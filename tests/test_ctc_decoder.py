import numpy as np
import pytest

from recognition.ctc_decoder import BLANK_INDEX, CTCCodec, OPTION_VALUE_CHARSET, PRICE_CHARSET, greedy_decode, prefix_beam_search


def matrix_for_path(indexes, class_count):
    matrix = np.full((len(indexes), class_count), 0.001, dtype=np.float64)
    for row, index in enumerate(indexes):
        matrix[row, index] = 0.99
        matrix[row] /= matrix[row].sum()
    return matrix


@pytest.mark.parametrize("text", ["+8", "+9%", "+130", "-2", "100", "110", "888"])
def test_option_value_ctc_encode_decode(text):
    codec = CTCCodec(OPTION_VALUE_CHARSET)
    path = []
    for index in codec.encode(text):
        path.extend([index, BLANK_INDEX])

    assert codec.decode_indices(path) == text
    assert greedy_decode(matrix_for_path(path, codec.num_classes), codec).text == text


@pytest.mark.parametrize("text", ["23,588,919", "1,299,999,999"])
def test_price_ctc_encode_decode(text):
    codec = CTCCodec(PRICE_CHARSET)
    path = []
    for index in codec.encode(text):
        path.extend([index, BLANK_INDEX])

    assert greedy_decode(matrix_for_path(path, codec.num_classes), codec).text == text


def test_repeated_characters_need_blank_to_separate():
    codec = CTCCodec(OPTION_VALUE_CHARSET)
    one = codec.char_to_index["1"]
    zero = codec.char_to_index["0"]

    assert codec.decode_indices([one, zero, zero]) == "10"
    assert codec.decode_indices([one, zero, BLANK_INDEX, zero]) == "100"


def test_beam_search_results_are_sorted():
    codec = CTCCodec(OPTION_VALUE_CHARSET)
    plus = codec.char_to_index["+"]
    eight = codec.char_to_index["8"]
    nine = codec.char_to_index["9"]
    matrix = np.full((2, codec.num_classes), 0.001, dtype=np.float64)
    matrix[0, plus] = 0.95
    matrix[1, eight] = 0.45
    matrix[1, nine] = 0.40
    matrix /= matrix.sum(axis=1, keepdims=True)

    candidates = prefix_beam_search(matrix, codec, beam_width=5, top_k=3)

    assert candidates[0].score >= candidates[1].score
    assert candidates[0].text == "+8"


def test_invalid_ctc_index_raises():
    codec = CTCCodec(OPTION_VALUE_CHARSET)

    with pytest.raises(ValueError):
        codec.decode_indices([999])
