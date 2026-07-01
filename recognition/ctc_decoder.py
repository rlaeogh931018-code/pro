from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


OPTION_VALUE_CHARSET = "0123456789+-%,"
PRICE_CHARSET = "0123456789,"
BLANK_INDEX = 0


@dataclass(frozen=True)
class CTCCandidate:
    text: str
    score: float


class CTCCodec:
    def __init__(self, charset: str) -> None:
        if len(set(charset)) != len(charset):
            raise ValueError("charset must not contain duplicate characters")
        self.charset = charset
        self.blank_index = BLANK_INDEX
        self.char_to_index = {char: index + 1 for index, char in enumerate(charset)}
        self.index_to_char = {index + 1: char for index, char in enumerate(charset)}

    @property
    def num_classes(self) -> int:
        return len(self.charset) + 1

    def encode(self, text: str) -> list[int]:
        indexes = []
        for char in text:
            if char not in self.char_to_index:
                raise ValueError(f"character {char!r} is not in charset")
            indexes.append(self.char_to_index[char])
        return indexes

    def decode_indices(self, indexes: Iterable[int]) -> str:
        text: list[str] = []
        previous = self.blank_index
        for index in indexes:
            if index == self.blank_index:
                previous = index
                continue
            if index < 0 or index >= self.num_classes:
                raise ValueError(f"CTC index out of range: {index}")
            if index != previous:
                text.append(self.index_to_char[index])
            previous = index
        return "".join(text)


def greedy_decode(logits_or_probs: Sequence[Sequence[float]], codec: CTCCodec) -> CTCCandidate:
    matrix = _as_matrix(logits_or_probs, codec.num_classes)
    indexes = np.argmax(matrix, axis=1).tolist()
    score = float(np.mean(np.max(_to_probabilities(matrix), axis=1))) if matrix.size else 0.0
    return CTCCandidate(codec.decode_indices(indexes), score)


def prefix_beam_search(
    logits_or_probs: Sequence[Sequence[float]],
    codec: CTCCodec,
    beam_width: int = 10,
    top_k: int = 3,
) -> list[CTCCandidate]:
    matrix = _as_matrix(logits_or_probs, codec.num_classes)
    log_probs = _to_log_probabilities(matrix)
    beams: dict[str, tuple[float, float]] = {"": (0.0, -math.inf)}
    for row in log_probs:
        next_beams: dict[str, tuple[float, float]] = {}
        for prefix, (p_blank, p_non_blank) in beams.items():
            for index, log_prob in enumerate(row):
                if index == codec.blank_index:
                    nb_blank, nb_non_blank = next_beams.get(prefix, (-math.inf, -math.inf))
                    nb_blank = _logsumexp(nb_blank, p_blank + log_prob, p_non_blank + log_prob)
                    next_beams[prefix] = (nb_blank, nb_non_blank)
                    continue
                char = codec.index_to_char.get(index)
                if char is None:
                    raise ValueError(f"CTC index out of range: {index}")
                end_char = prefix[-1:] if prefix else ""
                if char == end_char:
                    nb_blank, nb_non_blank = next_beams.get(prefix, (-math.inf, -math.inf))
                    nb_non_blank = _logsumexp(nb_non_blank, p_non_blank + log_prob)
                    next_beams[prefix] = (nb_blank, nb_non_blank)
                    extended = prefix + char
                    eb_blank, eb_non_blank = next_beams.get(extended, (-math.inf, -math.inf))
                    eb_non_blank = _logsumexp(eb_non_blank, p_blank + log_prob)
                    next_beams[extended] = (eb_blank, eb_non_blank)
                else:
                    extended = prefix + char
                    eb_blank, eb_non_blank = next_beams.get(extended, (-math.inf, -math.inf))
                    eb_non_blank = _logsumexp(eb_non_blank, p_blank + log_prob, p_non_blank + log_prob)
                    next_beams[extended] = (eb_blank, eb_non_blank)
        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: _logsumexp(item[1][0], item[1][1]),
                reverse=True,
            )[: max(1, beam_width)]
        )
    candidates = [
        CTCCandidate(text, math.exp(_logsumexp(p_blank, p_non_blank)))
        for text, (p_blank, p_non_blank) in beams.items()
    ]
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates[:top_k]


def _as_matrix(values: Sequence[Sequence[float]], expected_classes: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected 2D CTC matrix, got shape {matrix.shape}")
    if matrix.shape[1] != expected_classes:
        raise ValueError(f"expected {expected_classes} classes, got {matrix.shape[1]}")
    return matrix


def _to_probabilities(matrix: np.ndarray) -> np.ndarray:
    if np.all(matrix >= 0.0) and np.allclose(matrix.sum(axis=1), 1.0, atol=1e-4):
        return matrix
    shifted = matrix - matrix.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _to_log_probabilities(matrix: np.ndarray) -> np.ndarray:
    probs = _to_probabilities(matrix)
    return np.log(np.clip(probs, 1e-12, 1.0))


def _logsumexp(*values: float) -> float:
    finite = [value for value in values if value != -math.inf]
    if not finite:
        return -math.inf
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))
