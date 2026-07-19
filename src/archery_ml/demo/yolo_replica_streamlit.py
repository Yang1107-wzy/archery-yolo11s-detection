from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from archery_ml.demo.yolo_replica_demo import (
    ARCHERY_ML_ROOT,
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    EXPECTED_CHECKPOINT_SHA256,
    NO_GT_NOTICE,
    POINT_NOTICE,
    InferenceResult,
    annotated_png_bytes,
    build_batch_zip,
    decode_uploaded_image,
    duplicate_filenames,
    load_metrics_dashboard,
    predict_uploaded_image,
    result_csv_bytes,
    result_json_bytes,
    save_result_artifacts,
    validate_checkpoint,
)


DEFAULT_HOST = "::"
DEFAULT_PORT = 8504
PAGE_TABS = ("单张推理", "批量推理", "模型指标")
OUTPUT_ROOT = ARCHERY_ML_ROOT / "outputs" / "yolo_replica_web"
PAGE_TITLE = "YOLO11s 150-Epoch 箭靶检测可视化"
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "webp"]


def build_yolo_streamlit_command(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> list[str]:
    return [
        "streamlit",
        "run",
        "app/yolo_replica_demo.py",
        "--server.headless",
        "true",
        "--server.address",
        str(host),
        "--server.port",
        str(port),
    ]


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem.strip() or "image"
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", stem).strip("._")
    return cleaned or "image"


def _run_dir(prefix: str, digest: str) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_ROOT / f"{prefix}_{timestamp}_{digest[:8]}"


def _persist_single(result: InferenceResult, data: bytes) -> Path:
    digest = hashlib.sha256(data).hexdigest()
    output = _run_dir(f"single_{_safe_stem(result.image_name)}", digest)
    return save_result_artifacts(result, output)


def _batch_summary_rows(results: list[InferenceResult]) -> list[dict[str, Any]]:
    return [
        {
            "文件名": item.image_name,
            "全部检测": item.detection_count,
            "分数类检测": item.score_detection_count,
            "检测到箭靶": item.target_detected,
            "平均置信度": item.mean_confidence,
            "最高置信度": item.max_confidence,
            "耗时(ms)": item.latency_ms,
            "设备": item.device,
            "类别分布": json.dumps(item.class_counts, ensure_ascii=False),
        }
        for item in results
    ]


def _display_model_identity(st) -> None:
    try:
        actual_hash = validate_checkpoint(DEFAULT_CHECKPOINT)
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()
    st.success("正式模型身份校验通过")
    st.caption(
        "模型：YOLO11s Target v6 local replica｜训练：150 epochs / batch 8 / seed 42 / patience 0｜"
        "部署：best.pt（best epoch 142）"
    )
    st.code(f"checkpoint: {DEFAULT_CHECKPOINT}\nSHA-256: {actual_hash}", language="text")


def _threshold_controls(st) -> tuple[float, float]:
    st.sidebar.subheader("推理工作点")
    if "yolo_confidence" not in st.session_state:
        st.session_state.yolo_confidence = DEFAULT_CONFIDENCE
    if "yolo_iou" not in st.session_state:
        st.session_state.yolo_iou = DEFAULT_IOU
    if st.sidebar.button("恢复正式工作点"):
        st.session_state.yolo_confidence = DEFAULT_CONFIDENCE
        st.session_state.yolo_iou = DEFAULT_IOU
        st.session_state.confidence_slider = DEFAULT_CONFIDENCE
        st.session_state.iou_slider = DEFAULT_IOU
        st.rerun()
    confidence = st.sidebar.slider(
        "Confidence",
        min_value=0.05,
        max_value=0.95,
        value=float(st.session_state.yolo_confidence),
        step=0.05,
        key="confidence_slider",
    )
    iou = st.sidebar.slider(
        "NMS IoU",
        min_value=0.10,
        max_value=0.90,
        value=float(st.session_state.yolo_iou),
        step=0.05,
        key="iou_slider",
    )
    st.session_state.yolo_confidence = confidence
    st.session_state.yolo_iou = iou
    st.sidebar.caption(f"当前值：confidence={confidence:.2f}，IoU={iou:.2f}")
    return float(confidence), float(iou)


def _detection_rows(result: InferenceResult) -> list[dict[str, Any]]:
    return [
        {
            "ID": item.detection_id,
            "class_id": item.class_id,
            "label": item.label,
            "confidence": item.confidence,
            "x1": item.x1,
            "y1": item.y1,
            "x2": item.x2,
            "y2": item.y2,
            "center_x": item.center_x,
            "center_y": item.center_y,
            "width": item.width,
            "height": item.height,
        }
        for item in result.detections
    ]


def _render_result(st, result: InferenceResult, original_image, output_dir: Path | None = None) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("原始图片")
        st.image(original_image, channels="BGR", use_container_width=True)
    with right:
        st.subheader("模型预测")
        st.image(result.annotated_image, channels="BGR", use_container_width=True)

    cards = st.columns(6)
    cards[0].metric("全部检测", result.detection_count)
    cards[1].metric("分数类检测", result.score_detection_count)
    cards[2].metric("检测到箭靶", "是" if result.target_detected else "否")
    cards[3].metric("平均置信度", f"{result.mean_confidence:.3f}")
    cards[4].metric("最高置信度", f"{result.max_confidence:.3f}")
    cards[5].metric("推理耗时", f"{result.latency_ms:.1f} ms")
    st.caption(
        f"设备：{result.device}｜图片：{result.width}×{result.height}｜"
        f"confidence={result.confidence_threshold:.2f}｜IoU={result.iou_threshold:.2f}"
    )
    if result.device_fallback_reason:
        st.warning(f"MPS 推理失败，本次已回退 CPU：{result.device_fallback_reason}")

    rows = _detection_rows(result)
    st.subheader("检测明细")
    if not rows:
        st.info("当前工作点下没有检测结果。这是合法输出，未伪造任何框。")
    else:
        labels = sorted({row["label"] for row in rows})
        filter_cols = st.columns(2)
        selected = filter_cols[0].multiselect("表格类别筛选", labels, default=labels)
        minimum = filter_cols[1].slider(
            "表格最低置信度（不重新推理）", 0.0, 1.0, 0.0, 0.05, key=f"table_{result.image_name}"
        )
        filtered = [row for row in rows if row["label"] in selected and row["confidence"] >= minimum]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.write("类别统计：", result.class_counts)

    stem = _safe_stem(result.image_name)
    download_cols = st.columns(3)
    download_cols[0].download_button(
        "下载 result.json", result_json_bytes(result), f"{stem}_result.json", "application/json"
    )
    download_cols[1].download_button(
        "下载 detections.csv", result_csv_bytes(result), f"{stem}_detections.csv", "text/csv"
    )
    download_cols[2].download_button(
        "下载 annotated.png", annotated_png_bytes(result), f"{stem}_annotated.png", "image/png"
    )
    if output_dir is not None:
        st.caption(f"本次结果已保存：{output_dir}")


def _render_single_tab(st, confidence: float, iou: float) -> None:
    st.warning(NO_GT_NOTICE)
    st.info(POINT_NOTICE)
    upload = st.file_uploader("上传一张本地照片", type=SUPPORTED_TYPES, key="single_upload")
    if upload is None:
        st.info("请选择一张图片开始真实本地推理。")
        return
    data = upload.getvalue()
    cache_key = hashlib.sha256(data + f"{confidence:.3f}-{iou:.3f}".encode()).hexdigest()
    try:
        if st.session_state.get("single_cache_key") != cache_key:
            with st.spinner("正在使用正式 best.pt 推理..."):
                result = predict_uploaded_image(data, upload.name, confidence=confidence, iou=iou)
                output = _persist_single(result, data)
            st.session_state.single_cache_key = cache_key
            st.session_state.single_result = result
            st.session_state.single_output = output
        result = st.session_state.single_result
        original = decode_uploaded_image(data, upload.name)
    except Exception as exc:
        st.error(f"推理失败：{exc}")
        return
    _render_result(st, result, original, st.session_state.single_output)


def _render_batch_tab(st, confidence: float, iou: float) -> None:
    st.warning(NO_GT_NOTICE)
    uploads = st.file_uploader(
        "批量上传本地照片",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key="batch_uploads",
    )
    if not uploads:
        st.info("可一次选择多张图片；单张失败不会中断其他图片。")
        return
    duplicates = duplicate_filenames([item.name for item in uploads])
    if duplicates:
        st.warning(
            "检测到同名文件："
            + "、".join(duplicates)
            + "。结果会自动添加序号，确保不会相互覆盖。"
        )
    if st.button("开始批量推理", type="primary"):
        results: list[InferenceResult] = []
        errors: list[dict[str, str]] = []
        progress = st.progress(0.0)
        for index, upload in enumerate(uploads, start=1):
            try:
                results.append(
                    predict_uploaded_image(
                        upload.getvalue(), upload.name, confidence=confidence, iou=iou
                    )
                )
            except Exception as exc:
                errors.append({"文件名": upload.name, "错误": str(exc)})
            progress.progress(index / len(uploads))
        digest = hashlib.sha256("|".join(item.name for item in uploads).encode()).hexdigest()
        output = _run_dir("batch", digest)
        output.mkdir(parents=True, exist_ok=True)
        for index, result in enumerate(results, start=1):
            save_result_artifacts(result, output / f"{index:03d}_{_safe_stem(result.image_name)}")
        archive = build_batch_zip(results)
        (output / "batch_results.zip").write_bytes(archive)
        if errors:
            (output / "errors.json").write_text(
                json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        st.session_state.batch_results = results
        st.session_state.batch_errors = errors
        st.session_state.batch_archive = archive
        st.session_state.batch_output = output

    results = st.session_state.get("batch_results", [])
    errors = st.session_state.get("batch_errors", [])
    if results:
        st.subheader("批量汇总")
        st.dataframe(_batch_summary_rows(results), use_container_width=True, hide_index=True)
        st.download_button(
            "下载全部结果 ZIP",
            st.session_state.batch_archive,
            "yolo_replica_batch_results.zip",
            "application/zip",
        )
        st.caption(f"批量结果已保存：{st.session_state.batch_output}")
        for result in results:
            with st.expander(f"{result.image_name}｜{result.detection_count} 个检测"):
                st.image(result.annotated_image, channels="BGR", use_container_width=True)
                st.dataframe(_detection_rows(result), use_container_width=True, hide_index=True)
    if errors:
        st.subheader("失败文件")
        st.dataframe(errors, use_container_width=True, hide_index=True)


def _metric_value(container, label: str, value: Any) -> None:
    container.metric(label, "—" if value is None else f"{float(value):.5f}")


def _render_metrics_tab(st) -> None:
    st.info("本页指标来自固定数据集评测，不是当前上传图片的准确率。")
    try:
        dashboard = load_metrics_dashboard()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        st.error(str(exc))
        return

    st.subheader("Best validation（训练选模记录）")
    best = dashboard["best_validation"]
    best_cols = st.columns(4)
    _metric_value(best_cols[0], "Precision", best.get("metrics/precision(B)"))
    _metric_value(best_cols[1], "Recall", best.get("metrics/recall(B)"))
    _metric_value(best_cols[2], "mAP50", best.get("metrics/mAP50(B)"))
    _metric_value(best_cols[3], "mAP50-95", best.get("metrics/mAP50-95(B)"))

    selected = dashboard["selected"]
    st.subheader("Locked validation operating point")
    st.write(f"confidence={selected.get('confidence')}，IoU={selected.get('iou')}")

    public = dashboard["public_test"]
    st.subheader("Official public test（65 张）")
    public_cols = st.columns(5)
    for column, key, label in zip(
        public_cols,
        ("precision", "recall", "f1", "map50", "map50_95"),
        ("Precision", "Recall", "F1", "mAP50", "mAP50-95"),
    ):
        _metric_value(column, label, public.get(key))
    st.warning(
        "官方 split 存在相似图/同源候选，public-test 指标仍可能高估独立场景泛化；"
        "不要把它写成任意新照片的保证。"
    )

    st.subheader("12 类 public-test 指标")
    st.dataframe(dashboard["per_class"], use_container_width=True, hide_index=True)

    confusion = Path(dashboard["confusion_matrix_path"])
    training_graph = Path(dashboard["training_graph_path"])
    plots = st.columns(2)
    if confusion.is_file():
        plots[0].image(str(confusion), caption="归一化混淆矩阵", use_container_width=True)
    if training_graph.is_file():
        plots[1].image(str(training_graph), caption="150 epoch 训练曲线", use_container_width=True)

    st.subheader("逐 epoch 曲线数据")
    rows = dashboard["training_rows"]
    if rows:
        import pandas as pd

        frame = pd.DataFrame(rows).set_index("epoch")
        metric_columns = [
            key
            for key in (
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
            )
            if key in frame.columns
        ]
        loss_columns = [
            key
            for key in ("train/box_loss", "train/cls_loss", "train/dfl_loss", "val/box_loss", "val/cls_loss", "val/dfl_loss")
            if key in frame.columns
        ]
        st.line_chart(frame[metric_columns])
        st.line_chart(frame[loss_columns])
    st.caption(
        f"指标来源：{dashboard['metrics_path']}｜训练曲线：{dashboard['results_path']}｜"
        f"模型 hash：{EXPECTED_CHECKPOINT_SHA256}"
    )


def render_yolo_replica_app() -> None:
    try:
        import streamlit as st
    except Exception as exc:  # pragma: no cover - manual startup guard
        raise RuntimeError("缺少 Streamlit，无法启动 YOLO 可视化页面。") from exc

    st.set_page_config(page_title=PAGE_TITLE, page_icon="🎯", layout="wide")
    st.title(PAGE_TITLE)
    st.caption("本地上传、本地推理、本地保存；不会上传图片或修改训练权重。")
    _display_model_identity(st)
    confidence, iou = _threshold_controls(st)
    single_tab, batch_tab, metrics_tab = st.tabs(PAGE_TABS)
    with single_tab:
        _render_single_tab(st, confidence, iou)
    with batch_tab:
        _render_batch_tab(st, confidence, iou)
    with metrics_tab:
        _render_metrics_tab(st)
