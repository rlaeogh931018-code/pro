# Maple Auction Capture MVP

Windows desktop MVP for capturing a hovered Maple auction item, showing a review UI, and saving the final values to SQLite.

This version uses OpenCV template matching for fixed labels, values, and prices.
EasyOCR is optional and disabled by default. Phase 3 adds optional PyTorch model
structures for future training and inference, but missing checkpoints fall back
to the existing template path.

## Stack

- Python 3.12
- PySide6
- MSS
- pywin32
- pynput
- OpenCV
- SQLite
- pytest

Optional ML:

- PyTorch
- torchvision
- scikit-learn

## Run

Double-click:

```powershell
run_price_tool.bat
```

Or run manually:

```powershell
python app.py
```

## Workflow

1. Start the game in window mode.
2. Make sure the game window title contains the keyword in `config.yaml`.
3. Run the app.
4. Hover the mouse over an auction item.
5. Press `F8`.
6. The app finds the game window, checks that the mouse is inside the client area, captures the configured mouse-relative region, and saves only the PNG.
7. Select a PNG from the capture dropdown.
8. Click `Analyze Selected PNG`.
9. Edit fields using the keyboard if needed.
10. Press `Enter` or `Ctrl+S` to save.
11. Press `Esc` to cancel.

EasyOCR fallback is disabled by default with `enable_easyocr_fallback: false`.

## Saved Data

SQLite database:

```text
ITEMDB\auction_records.sqlite3
```

Saved columns include:

- final values
- raw recognized values
- confidence values
- image path
- capture timestamp
- save timestamp

The app warns when a similar record was saved recently.

## Capture Config

Edit `config.yaml`:

```yaml
capture:
  left: 500
  right: 900
  up: 100
  down: 250
```

These values define the asymmetric capture region around the mouse.

You can also change them at runtime:

```text
Settings -> Capture Region...
```

- `Apply Runtime`: applies the values until the app closes.
- `Save to config.yaml`: applies and persists the values.

## Templates

Template root:

```text
templates\
```

Required structure:

```text
templates\
  option_labels\
    REQ LEV.png
    장비분류.png
    str.png
    dex.png
    int.png
    luk.png
    attack.png
    마력.png
    업그레이드가능횟수.png
    black_crystal.png
    잠재_공격시 몬스터의 방어율.png
    잠재_마력.png
    잠재_보스공격시.png
  equipment_types\
    wand.png
    staff.png
```

Create option label templates by cropping only the static label text from a clean tooltip screenshot, for example `INT :`, `마력 :`, or `업그레이드 가능 횟수 :`.

Numbers and prices are currently read by OpenCV templates and digit fallback
logic. Optional CRNN checkpoints can be added later under `models\`.

Thresholds and OCR options live in `config.yaml`:

```yaml
vision:
  option_threshold: 0.82
  alignment_enabled: true
  ml_enabled: true
  device: auto
  option_classifier_checkpoint: models/option_classifier.pt
  option_classifier_pretrained: false
  option_value_crnn_checkpoint: models/option_value_crnn.pt
  price_crnn_checkpoint: models/price_crnn.pt
  ctc_beam_width: 10
  ctc_top_k: 3
  template_fallback_enabled: true
  enable_easyocr_fallback: false
  easyocr_languages:
    - ko
    - en
  easyocr_gpu: false
```

## Optional ML Setup

Runtime dependencies stay in `requirements.txt`. ML dependencies are separate:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

For CUDA builds of PyTorch, use the official PyTorch install selector for your
driver/CUDA environment instead of hardcoding a wheel URL in this repository.

Training data is expected as JSONL plus PNG crops:

```text
datasets\
  option_labels\
    images\
    samples.jsonl
  option_values\
    images\
    samples.jsonl
  prices\
    images\
    samples.jsonl
```

Each JSONL row includes `image_path`, `label`, `session_id`, `field_type`, and
`was_corrected`.

Collection verification and preview entry points:

```powershell
.\.venv\Scripts\python.exe -m training.replay_capture --before captures\before_YYYYMMDD_HHMMSS_x.png --after captures\after_YYYYMMDD_HHMMSS_x.png --confirmed-values confirmed_values.json
.\.venv\Scripts\python.exe -m training.inspect_dataset --dataset-dir datasets --show-invalid --export-report inspect_report.json --export-html inspect_report.html
.\.venv\Scripts\python.exe -m training.preview_dataset --dataset-dir datasets --task option_value --limit 100 --output preview_option_value.html
.\.venv\Scripts\python.exe -m training.review_dataset --config config.yaml --task price
.\.venv\Scripts\python.exe -m training.clean_dataset --config config.yaml --task price --dry-run
```

`replay_capture` defaults to temporary DB, dataset, and debug paths when those
paths are not provided, so it can smoke-test the end-to-end collection path
without writing into the operational database or dataset.
Training datasets include only `approved` samples by default; newly collected
samples are `unreviewed` until `review_dataset` marks them approved.

Training and evaluation entry points:

```powershell
.\.venv\Scripts\python.exe -m training.train_option_classifier --config config.yaml
.\.venv\Scripts\python.exe -m training.train_crnn --task option_value --config config.yaml
.\.venv\Scripts\python.exe -m training.train_crnn --task price --config config.yaml
.\.venv\Scripts\python.exe -m training.evaluate_models --config config.yaml
```

## Current Modules

- `maple_price_tool.capture`: game window discovery and screen capture
- `maple_price_tool.vision`: OpenCV template matching plus optional ML fallback traces
- `recognition`: alignment, preprocessing, CTC decoding, optional model registry and datasets
- `training`: optional ML training/evaluation CLIs
- `maple_price_tool.ui`: PySide6 review UI and F8 hotkey wiring
- `maple_price_tool.storage`: SQLite persistence and duplicate detection
- `maple_price_tool.domain`: state, DTOs, and shared types
- `maple_price_tool.config`: YAML config loader

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
