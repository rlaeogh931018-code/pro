from __future__ import annotations

from dataclasses import dataclass

from .ctc_decoder import OPTION_VALUE_CHARSET, PRICE_CHARSET


try:  # pragma: no cover - optional ML dependency.
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass(frozen=True)
class CRNNConfig:
    charset: str
    input_channels: int = 3
    hidden_size: int = 256
    lstm_layers: int = 2
    dropout: float = 0.1

    @property
    def num_classes(self) -> int:
        return len(self.charset) + 1


def option_value_crnn_config() -> CRNNConfig:
    return CRNNConfig(charset=OPTION_VALUE_CHARSET)


def price_crnn_config() -> CRNNConfig:
    return CRNNConfig(charset=PRICE_CHARSET)


if nn is not None:

    class CRNN(nn.Module):
        def __init__(self, config: CRNNConfig) -> None:
            super().__init__()
            self.config = config
            self.cnn = nn.Sequential(
                nn.Conv2d(config.input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),
            )
            self.lstm = nn.LSTM(
                input_size=256,
                hidden_size=config.hidden_size,
                num_layers=config.lstm_layers,
                dropout=config.dropout if config.lstm_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.classifier = nn.Linear(config.hidden_size * 2, config.num_classes)
            self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

        def forward(self, images):
            features = self.cnn(images)
            features = features.mean(dim=2)
            sequence = features.permute(2, 0, 1)
            recurrent, _hidden = self.lstm(sequence)
            return self.classifier(recurrent)

        def sequence_lengths(self, input_widths):
            return (input_widths // 4).clamp(min=1)

else:

    class CRNN:  # type: ignore[no-redef]
        def __init__(self, _config: CRNNConfig) -> None:
            raise RuntimeError("torch is required for CRNN models. Install requirements-ml.txt.")
