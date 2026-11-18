
import torch
import torch.nn as nn
from pathlib import Path


def conv_block(in_channels, out_channels):
    """
    Two convolution layers followed by max pooling.
    Same architecture used during training.
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),

        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2)
    )


class CatDogCNN(nn.Module):

    def __init__(self):
        super(CatDogCNN, self).__init__()

        self.conv_block1 = conv_block(3, 32)
        self.conv_block2 = conv_block(32, 64)
        self.conv_block3 = conv_block(64, 128)
        self.conv_block4 = conv_block(128, 256)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = self.global_pool(x)
        x = self.classifier(x)

        return x


def load_model(checkpoint_path, device=None):

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = CatDogCNN().to(device)

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        # Compatibility for PyTorch releases before weights_only was added.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Invalid checkpoint format: expected a dictionary.")

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Invalid checkpoint format: 'model_state_dict' is missing."
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model, checkpoint
