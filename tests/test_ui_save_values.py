from __future__ import annotations

import pytest

from maple_price_tool.ui import parse_optional_int, parse_required_int


def test_optional_numeric_fields_default_to_zero_when_blank():
    assert parse_optional_int("") == 0
    assert parse_optional_int("  ") == 0
    assert parse_optional_int("None") == 0


def test_numeric_parsing_accepts_commas():
    assert parse_required_int("1,111,111") == 1111111
    assert parse_optional_int("1,234") == 1234


def test_required_numeric_field_rejects_blank():
    with pytest.raises(ValueError):
        parse_required_int("")
