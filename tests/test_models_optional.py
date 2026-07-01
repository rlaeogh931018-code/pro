import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from recognition.crnn import CRNN, option_value_crnn_config, price_crnn_config
from recognition.model_registry import ModelRegistry
from recognition.option_classifier import build_option_classifier, default_option_class_names
from maple_price_tool.config import VisionConfig


def test_option_classifier_forward_shape():
    class_names = default_option_class_names()
    model = build_option_classifier(len(class_names), pretrained=False)
    output = model(torch.zeros(2, 3, 32, 256))

    assert output.shape == (2, len(class_names))


def test_crnn_forward_shape():
    torch.backends.cudnn.enabled = False
    config = option_value_crnn_config()
    model = CRNN(config)
    output = model(torch.zeros(2, 3, 32, 192))

    assert output.shape[1] == 2
    assert output.shape[2] == config.num_classes


def test_price_crnn_forward_and_ctc_backward_cuda_if_available():
    torch.backends.cudnn.enabled = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = price_crnn_config()
    model = CRNN(config).to(device)
    logits = model(torch.zeros(2, 3, 32, 384, device=device))
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    input_lengths = torch.full((2,), logits.shape[0], dtype=torch.long, device=device)
    targets = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long, device=device)
    target_lengths = torch.tensor([2, 3], dtype=torch.long, device=device)
    loss = model.ctc_loss(log_probs, targets, input_lengths, target_lengths)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.classifier.weight.grad is not None


def test_crnn_checkpoint_roundtrip_and_registry_lazy_loading(tmp_path):
    config = option_value_crnn_config()
    model = CRNN(config)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": "crnn",
        "task": "option_value",
        "charset": config.charset,
        "preprocessing_config": {"target_height": 32, "channels": 3},
        "model_config": config.__dict__,
        "epoch": 1,
        "validation_metrics": {},
        "created_at": "test",
    }
    path = tmp_path / "option_value_crnn.pt"
    torch.save(checkpoint, path)
    registry = ModelRegistry(VisionConfig(option_value_crnn_checkpoint=path, device="cpu"))

    first = registry.get_option_value_crnn()
    second = registry.get_option_value_crnn()

    assert first is second
    assert first is not None
    assert registry.load_counts["option_value_crnn"] == 1
