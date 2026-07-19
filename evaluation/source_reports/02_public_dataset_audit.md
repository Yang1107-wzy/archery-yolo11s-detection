# Public Dataset v6 audit

Status: **blocked before data audit**.

No Target and Arrow Detection v6 ZIP or `data/raw/target_arrow_v6/data.yaml` was found locally. `ARCHERY_ALLOW_NETWORK` and `ROBOFLOW_API_KEY` are unset, and the Roboflow SDK/CLI is not installed. The audit command correctly returned `missing_data_yaml`; no image count, split, class mapping, license, duplicate, leakage, or public-data fingerprint is claimed.

Expected public reference values remain unverified locally: 1,645 images with 1,482 train / 98 valid / 65 test. Actual class order must come from the downloaded `data.yaml`.

The isolated synthetic fixture audit passed for 32 train / 16 valid / 8 test images, twelve classes, 56 target boxes, 56 score boxes, zero audit issues, and fingerprint `91a7ab772e1ece421780ef1a544cedf54ad8afda0fd4f0164464c26e834834a5`. This fixture is plumbing-only and is not Dataset v6.
