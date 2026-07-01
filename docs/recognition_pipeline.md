# Recognition Pipeline

This document captures the current Maple auction recognition flow before the ML
recognition package is introduced.

## Capture Flow

1. `ReviewWindow` in `maple_price_tool/ui.py` listens for global hotkeys.
2. `F7` calls `start_before_capture()` and captures the whole game client with
   prefix `before`.
3. `F8` calls `start_after_capture()` and captures the whole game client with
   prefix `after`.
4. When an `after` capture has a matching `before` image, `write_before_sidecar()`
   writes `<after>.before.txt`. Later analysis reads that path through
   `read_before_sidecar()`.

## Analysis Entry Point

1. `AnalyzeImageWorker.run()` builds a `CaptureResult`.
2. It creates `OpenCvTemplateRecognizer` from `maple_price_tool/vision.py`.
3. `OpenCvTemplateRecognizer.analyze()` loads the after image and calls
   `analyze_maple_layout()`.

## Tooltip Detection

Primary functions:

- `find_yellow_tooltip_rect()` in `maple_price_tool/vision.py`
- `find_diff_tooltip_rect()` in `maple_price_tool/vision.py`

The recognizer first tries to find the yellow tooltip directly in the after
image. If that fails and a before image exists, it compares before/after to infer
the tooltip rectangle.

## Residual Generation

Primary functions:

- `build_diff_line_mask()`
- `build_diff_foreground_mask()`
- `fit_translucent_background()`
- `calculate_auto_threshold()`

`build_diff_line_mask()` loads the before image, validates dimensions, and passes
the before/after pair to `build_diff_foreground_mask()`. The residual algorithm
fits the translucent tooltip background from the before image and creates debug
images including `residual`, `analysis_binary`, `final_mask`, and
`foreground_text_mask`.

## Line Extraction

Primary functions:

- `read_tooltip_line_analysis()`
- `extract_tooltip_lines()`
- `extract_mask_lines()`

When a diff mask exists, line extraction uses the mask. Otherwise it falls back
to text color masks and template locations. The current line extraction is still
OpenCV-based and uses horizontal projection over foreground pixels.

## Option Name, Value, And Price Recognition

Primary functions:

- option labels: `match_option_line_label()`, `match_potential_line_label()`
- whole-line values: `match_known_option_line()`, `match_known_potential_line()`
- value templates: `match_value_pattern()`, `match_best_value_pattern()`
- digit fallback: `split_maple_character_groups()`, `classify_maple_digit()`
- price: `read_maple_price()`, `match_price_pattern()`,
  `recognize_maple_price_digits()`

The current system primarily uses template matching. Character-by-character
digit recognition remains as fallback/comparison logic. EasyOCR was previously
initialized eagerly for generic number reading; Phase 2 changes it to an
explicit lazy fallback only.

## UI Review

`ReviewWindow.on_analysis_ready()` populates editable fields from
`AnalysisResult.editable_values()`. The user can correct values before saving.
The F7/F8, Analyze, Review, and Save flows remain in `maple_price_tool/ui.py`.

## DB Storage

`ReviewWindow.save_current()` calls `final_record_from_analysis()` in
`maple_price_tool/storage.py`, then `Storage.save()` persists the final record to
SQLite. The current schema stores final values, raw recognized values,
confidences, image path, capture timestamp, and save timestamp.
