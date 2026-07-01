# Recognition Technical Debt

This document lists recognition risks found before Phase 2 changes.

## Hardcoded Verified Rows

Resolved in Phase 3 for production code:

- `VERIFIED_ROW_VALUES` was removed from `maple_price_tool/vision.py`.
- `analyze_verified_row()` was removed from `maple_price_tool/vision.py`.
- The old row/price expected values live only in
  `tests/fixtures/verified_rows.json`.

Production analysis must not return values from row/price identity.

## Silent Value Corrections

`normalize_potential_value()` in `maple_price_tool/vision.py` silently changes
recognized values. Examples include:

- boss/ignore defense values around `100..200`
- `80 -> 30`
- `5 -> 15`
- `8 -> 6`
- prefix stripping from values such as `119`

Phase 2 keeps behavior for compatibility but adds a traced helper so future
CRNN/constrained decoding can preserve raw and corrected predictions.

## Failure Converted To Zero

Several analysis paths convert failed recognition to `0` with `value or 0` or
dictionary defaults:

- `OpenCvTemplateRecognizer.analyze()`
- `OpenCvTemplateRecognizer.analyze_maple_layout()`
- `OpenCvTemplateRecognizer.build_price_only_result()`
- legacy `OpenCvTemplateRecognizer.analyze_verified_row()` paths removed in Phase 3

This can hide the difference between real `0` and failed recognition. Phase 2
starts preserving `None` on safer paths while keeping DB save compatibility for
confirmed user values.

## EasyOCR Eager Dependency

- module import block in `maple_price_tool/vision.py`
- `OpenCvTemplateRecognizer.__init__()`
- `OpenCvTemplateRecognizer.get_easyocr_reader()`
- `requirements.txt`

EasyOCR was previously a required runtime dependency and was created while
constructing the recognizer. Phase 2 makes it optional and lazy behind
`enable_easyocr_fallback`.

## Whole-Value Template Dependence

The current recognizer contains many fixed value templates:

- `VALUE_PATTERN_NAMES`
- `PRICE_PATTERN_NAMES`
- `match_value_pattern()`
- `match_best_value_pattern()`
- `match_price_pattern()`
- `match_known_option_line()`
- `match_known_potential_line()`

These are useful fallbacks and validation signals, but they are not scalable.
Future phases should keep them as fallback/fusion inputs while adding MobileNet
option classification and CRNN/CTC value recognition.
