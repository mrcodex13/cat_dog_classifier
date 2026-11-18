try:
    import spaces  # must be imported before torch on Hugging Face Spaces
except ImportError:
    pass

from pathlib import Path
import torch
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps
from torchvision import transforms

from model import load_model


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_cat_dog_model.pth"

# Force CPU: the model is small, and ZeroGPU Spaces only allow real GPU
# access inside @spaces.GPU functions, which this app does not need.
DEVICE = torch.device("cpu")

CLASS_NAMES = ["Cat", "Dog"]

IMG_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

model, checkpoint = load_model(MODEL_PATH, DEVICE)

print("Model loaded successfully!")
print("Device:", DEVICE)
print("Best Epoch:", checkpoint.get("epoch", "unknown"))


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

inference_transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(IMG_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=MEAN,
        std=STD
    )
])


# ============================================================
# 4. MODEL PREDICTION
# ============================================================

def predict_image(image):

    if image is None:
        return None, "Please upload an image first."

    try:

        # Handle image orientation and color format
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        # Apply exactly the validation/test preprocessing
        input_tensor = inference_transform(image)

        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(DEVICE)

        # Inference
        with torch.inference_mode():

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[0]

        probabilities = probabilities.cpu().numpy()

        cat_confidence = float(probabilities[0])
        dog_confidence = float(probabilities[1])

        predicted_index = int(np.argmax(probabilities))

        predicted_class = CLASS_NAMES[predicted_index]

        confidence = float(probabilities[predicted_index])

        # Gradio Label component format
        results = {
            "Cat": cat_confidence,
            "Dog": dog_confidence
        }

        details = f"""
### Prediction Result

**Predicted Class:** {predicted_class}

**Confidence:** {confidence * 100:.2f}%

| Class | Confidence |
|---|---:|
| Cat | {cat_confidence * 100:.2f}% |
| Dog | {dog_confidence * 100:.2f}% |

*Confidence is the model's softmax output, not a guarantee
that the prediction is correct.*
"""

        return results, details

    except Exception as error:

        return None, f"Prediction error: {str(error)}"


# ============================================================
# 5. TRAINING HISTORY
# ============================================================

history = checkpoint.get("history", {})
if not isinstance(history, dict):
    history = {}


def create_training_graphs():

    required = [
        "train_accuracy",
        "val_accuracy",
        "train_loss",
        "val_loss"
    ]

    if not all(
        isinstance(history.get(key), (list, tuple))
        and len(history[key]) > 0
        for key in required
    ):
        return None

    train_accuracy = history["train_accuracy"]
    val_accuracy = history["val_accuracy"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    accuracy_epochs = range(
        1,
        max(len(train_accuracy), len(val_accuracy)) + 1
    )
    loss_epochs = range(
        1,
        max(len(train_loss), len(val_loss)) + 1
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Accuracy graph
    axes[0].plot(
        range(1, len(train_accuracy) + 1),
        train_accuracy,
        marker="o",
        markersize=3,
        label="Training Accuracy"
    )

    axes[0].plot(
        range(1, len(val_accuracy) + 1),
        val_accuracy,
        marker="o",
        markersize=3,
        label="Validation Accuracy"
    )

    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(list(accuracy_epochs))
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Loss graph
    axes[1].plot(
        range(1, len(train_loss) + 1),
        train_loss,
        marker="o",
        markersize=3,
        label="Training Loss"
    )

    axes[1].plot(
        range(1, len(val_loss) + 1),
        val_loss,
        marker="o",
        markersize=3,
        label="Validation Loss"
    )

    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_xticks(list(loss_epochs))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()

    return fig


training_graphs = create_training_graphs()


# ============================================================
# 6. MODEL INFORMATION
# ============================================================

best_epoch = checkpoint.get("epoch", "Unavailable")
validation_accuracy = checkpoint.get("val_accuracy")
validation_loss = checkpoint.get("val_loss")


def format_metric(value, precision):
    if value is None:
        return "Unavailable"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def last_history_value(name):
    values = history.get(name)
    if isinstance(values, (list, tuple)) and values:
        return format_metric(values[-1], 2)
    return "Unavailable"


validation_accuracy_text = format_metric(validation_accuracy, 2)
validation_loss_text = format_metric(validation_loss, 4)

# Values obtained from your notebook's test evaluation.
TEST_ACCURACY = 96.91
TEST_LOSS = 0.1308
ROC_AUC = 0.9953

PARAMETERS = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


# ============================================================
# 7. MODEL CONFIGURATION DASHBOARD
# ============================================================

MODEL_INFO = f"""
# Model Overview

A custom Convolutional Neural Network (CNN) trained from scratch
to classify images into two categories: Cats and Dogs.

### Performance Metrics

| Metric | Result |
|---|---:|
| Test Accuracy | **{TEST_ACCURACY:.2f}%** |
| Validation Accuracy | **{validation_accuracy_text}%** |
| Training Accuracy | **{last_history_value('train_accuracy')}%** |
| Test Loss | **{TEST_LOSS:.4f}** |
| Validation Loss | **{validation_loss_text}** |
| ROC-AUC | **{ROC_AUC:.4f}** |
| Best Epoch | **{best_epoch}** |
| Trainable Parameters | **{PARAMETERS:,}** |

Test evaluation was performed on 3,750 images.

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Cat | 96.46% | 97.39% | 96.92% |
| Dog | 97.36% | 96.43% | 96.89% |

---

# Training Configuration

### Dataset

- Dataset: Cat and Dog Classification Dataset
- Train / Validation / Test split: 70% / 15% / 15%
- Random state: 42
- Stratified splitting
- Test images: 3,750

### Image Preprocessing

- Image dimensions: 224 × 224
- RGB conversion
- Resize: 256
- Center Crop: 224 × 224
- Normalization: ImageNet mean and standard deviation

### Data Augmentation

- Random Resized Crop
- Random Horizontal Flip (0.5)
- Random Affine (10 degrees)
- Translation: 0.08
- Scale: 0.95–1.05
- Color Jitter
- Random Erasing (0.2)

### CNN Architecture

| Layer | Output Channels |
|---|---:|
| Convolution Block 1 | 32 |
| Convolution Block 2 | 64 |
| Convolution Block 3 | 128 |
| Convolution Block 4 | 256 |
| Global Average Pooling | 256 |
| Dropout | 0.3 |
| Fully Connected | 2 |

Each convolution block contains two Conv2D layers,
Batch Normalization, ReLU and Max Pooling.

### Training Hyperparameters

| Parameter | Configuration |
|---|---|
| Epochs | 40 |
| Batch Size | 64 |
| Optimizer | AdamW |
| Initial Learning Rate | 0.001 |
| Weight Decay | 0.01 |
| Loss Function | Cross Entropy |
| Label Smoothing | 0.1 |
| Gradient Clipping | 1.0 |
| Early Stopping Patience | 10 |
| Scheduler | ReduceLROnPlateau |
| Scheduler Factor | 0.5 |
| Scheduler Patience | 3 |
| MixUp / CutMix | Disabled |
| Mixed Precision | Enabled during CUDA training |

### Hardware

Training hardware: NVIDIA Tesla T4 GPUs on Kaggle.

Inference device: {DEVICE}.

---

**Important:** Test and validation metrics describe performance on
your evaluation datasets. Softmax confidence on an uploaded image
is not the same thing as measured model accuracy.
"""


# ============================================================
# 8. GRADIO INTERFACE
# ============================================================

CUSTOM_CSS = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
}

#main-title {
    text-align: center;
}

.metric-note {
    text-align: center;
    opacity: 0.75;
}
"""


with gr.Blocks(title="Cat vs Dog Classifier") as demo:

    gr.Markdown(
        """
        # 🐱 Cat vs Dog Classifier 🐶

        ### An image classification model built using a custom CNN.

        Upload an image of a cat or dog and let the model predict
        its class along with the confidence scores.
        """,
        elem_id="main-title"
    )

    with gr.Tab("Image Classifier"):

        gr.Markdown(
            "## Test the Model\n"
            "Upload an image to get a prediction."
        )

        with gr.Row():

            with gr.Column():

                input_image = gr.Image(
                    type="pil",
                    label="Upload a Cat or Dog Image"
                )

                predict_button = gr.Button(
                    "Predict Image",
                    variant="primary"
                )

                clear_button = gr.Button("Clear")

            with gr.Column():

                prediction_output = gr.Label(
                    label="Prediction Confidence",
                    num_top_classes=2
                )

                details_output = gr.Markdown(
                    "Your prediction will appear here."
                )

        predict_button.click(
            fn=predict_image,
            inputs=input_image,
            outputs=[
                prediction_output,
                details_output
            ]
        )

        clear_button.click(
            fn=lambda: (None, None, "Your prediction will appear here."),
            inputs=None,
            outputs=[
                input_image,
                prediction_output,
                details_output
            ]
        )

        gr.Markdown(
            """
            **Note:** This model is trained specifically for cat and
            dog classification. It may still classify unrelated images
            as one of these two classes.
            """
        )

    with gr.Tab("Model Performance"):

        gr.Markdown(
            "## Model Evaluation and Performance"
        )

        gr.Markdown(MODEL_INFO)

    with gr.Tab("Training Analysis"):

        gr.Markdown(
            """
            ## Training History

            These graphs show how the model learned over 40 epochs.
            """
        )

        if training_graphs is not None:

            gr.Plot(
                value=training_graphs,
                label="Training Performance"
            )

        else:

            gr.Markdown(
                "Training history is unavailable in the checkpoint."
            )

    
    with gr.Tab("About the Model"):

        gr.Markdown(
            """
# 🐱🐶 About This Project

## 1. Project Overview

This project implements a custom Convolutional Neural Network
(CNN) built from scratch using PyTorch to classify images
into two categories: Cats and Dogs.

Unlike transfer-learning approaches, this model learns
visual features directly from the training dataset without
using pretrained weights.

The objective was to build a CNN that achieves high
classification accuracy while maintaining good
generalization on unseen images.

---

## 2. Dataset Preparation

The model was trained using a balanced dataset containing
**24,998 images**.

| Dataset Split | Images | Percentage |
|---|---:|---:|
| Training | 17,498 | 70% |
| Validation | 3,750 | 15% |
| Testing | 3,750 | 15% |
| **Total** | **24,998** | **100%** |

### Class Distribution

- Cats: 12,499 images
- Dogs: 12,499 images

### Data Splitting Strategy

I used stratified train-validation-test splitting
with a random state of 42.

This ensured that:
- Both classes remained equally represented.
- No image appeared in multiple dataset splits.
- The test dataset remained separate from training.

---

## 3. Image Preprocessing & Augmentation

One important part of improving classification performance
was introducing variations into the training images.

### Image Preprocessing

- RGB image conversion
- Image resizing and cropping
- Final image dimensions: 224 × 224
- ImageNet normalization

Normalization parameters:

Mean: [0.485, 0.456, 0.406]

Standard Deviation: [0.229, 0.224, 0.225]

### Data Augmentation Techniques

| Technique | Configuration | Purpose |
|---|---|---|
| Random Resized Crop | Scale 0.80–1.0 | Variation in object size |
| Horizontal Flip | 50% probability | Orientation variation |
| Random Affine | 10° rotation | Geometric variation |
| Translation | 8% | Object position variation |
| Scaling | 0.95–1.05 | Scale variation |
| Color Jitter | 0.2 | Lighting and color variation |
| Random Erasing | 20% probability | Occlusion robustness |

These transformations helped the model learn
features that were less dependent on a specific
image orientation, position or lighting condition.

Validation and test images were not augmented.
They received only resizing, center cropping and normalization.

---

## 4. CNN Architecture

The model was designed from scratch using four
convolutional blocks.

| Layer | Channels |
|---|---:|
| Input | 3 |
| Convolution Block 1 | 32 |
| Convolution Block 2 | 64 |
| Convolution Block 3 | 128 |
| Convolution Block 4 | 256 |
| Global Average Pooling | 256 |
| Fully Connected Layer | 2 |

### Inside Each Convolution Block

Each block consists of:

1. Conv2D (3 × 3 kernel)
2. Batch Normalization
3. ReLU Activation
4. Conv2D (3 × 3 kernel)
5. Batch Normalization
6. ReLU Activation
7. Max Pooling (2 × 2)

### Final Classification Head

After feature extraction:

- Adaptive Average Pooling
- Flattening
- Dropout (0.3)
- Fully Connected Layer
- Two output logits (Cat and Dog)

The final logits are converted into probabilities
using Softmax during inference.

---

## 5. Training Strategy & Optimization

The model was trained for 40 epochs using
two NVIDIA Tesla T4 GPUs on Kaggle.

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Framework | PyTorch |
| Epochs | 40 |
| Batch Size | 64 |
| Optimizer | AdamW |
| Initial Learning Rate | 0.001 |
| Weight Decay | 0.01 |
| Loss Function | Cross Entropy |
| Label Smoothing | 0.1 |
| Gradient Clipping | 1.0 |
| Early Stopping Patience | 10 |
| Mixed Precision | Enabled |
| Hardware | 2 × Tesla T4 |

### Optimization Techniques

**AdamW Optimizer**

Used adaptive gradient optimization with weight decay
to improve training stability and regularization.

**Label Smoothing**

Used a smoothing factor of 0.1 to discourage
excessively confident predictions during training.

**Gradient Clipping**

Limited gradient norm to 1.0 to help prevent
unstable gradient updates.

**ReduceLROnPlateau**

Reduced the learning rate by a factor of 0.5
when validation loss stopped improving for
three consecutive epochs.

**Early Stopping**

Configured patience of 10 epochs to help
avoid unnecessary training and overfitting.

**Automatic Mixed Precision (AMP)**

Used FP16 mixed precision during GPU training
to improve computational efficiency.

MixUp and CutMix were explored in the training
configuration but were disabled in the final run.

---

## 6. Final Model Performance

# 96.91% Test Accuracy

The final checkpoint was evaluated on a separate
test dataset containing 3,750 previously unseen images.

| Evaluation Metric | Result |
|---|---:|
| Training Accuracy | 98.06% |
| Validation Accuracy | 97.01% |
| Test Accuracy | 96.91% |
| Test Loss | 0.1308 |
| ROC-AUC | 0.9953 |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Cat | 96.46% | 97.39% | 96.92% |
| Dog | 97.36% | 96.43% | 96.89% |
| Macro Average | 96.91% | 96.91% | 96.91% |

### Test Set Predictions

Out of 3,750 test images:

- Correct predictions: 3,634
- Incorrect predictions: 116

### Confusion Matrix

| Actual / Predicted | Cat | Dog |
|---|---:|---:|
| Cat | 1,826 | 49 |
| Dog | 67 | 1,808 |

---

## 7. How Did the Model Achieve 96.91%?

The result was achieved through a combination
of several design and training decisions:

**1. Balanced Dataset**

Equal representation of cats and dogs helped
prevent class imbalance during training.

**2. Data Augmentation**

Random transformations exposed the model to
different appearances of the same class.

**3. Progressive Feature Extraction**

The four convolutional blocks progressively
learned more complex visual patterns, from
basic edges and textures to higher-level features.

**4. Regularization**

Batch Normalization, Dropout, Weight Decay
and Label Smoothing helped control overfitting.

**5. Learning Rate Scheduling**

ReduceLROnPlateau adjusted the learning rate
based on validation loss, allowing more refined
optimization during later training stages.

**6. Independent Evaluation**

The final accuracy was measured on a separate
test set that was not used to update model weights.

### Generalization Analysis

Training Accuracy: 98.06%

Validation Accuracy: 97.01%

Generalization Gap: 1.05 percentage points.

The relatively small gap indicates that the model
maintained similar performance on training and
validation data.

---

## 8. Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Scikit-learn
- Matplotlib
- Gradio
- Kaggle GPU Environment

### Project Highlights

- Custom CNN architecture built from scratch.
- Approximately 25,000 balanced images.
- Four convolutional blocks.
- Advanced image augmentation.
- Mixed-precision GPU training.
- Independent test evaluation.
- Interactive image classification interface.

---

## 9. Limitations & Future Improvements

Although the model performs well on cat and dog images,
it has some limitations.

- It is a binary classifier and does not recognize unrelated objects.
- Softmax confidence is not a guarantee of correctness.
- Performance may vary on images with unusual lighting,
  backgrounds, or unfamiliar animal breeds.

Potential future improvements:

- Unknown-class detection.
- Model calibration.
- Grad-CAM visualization for model interpretability.
- Larger and more diverse datasets.
- Mobile deployment and inference optimization.

---

**Built with PyTorch | Trained on Kaggle | Deployed using Gradio**

*An end-to-end deep learning project exploring image
preprocessing, CNN architecture design, model optimization
and real-world inference.*
            """
        )


    gr.Markdown(
        """
        ---

        <div class="metric-note">
        Built with PyTorch and Gradio | Cat vs Dog Image Classification
        </div>
        """
    )


# ============================================================
# 9. LAUNCH APPLICATION
# ============================================================

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
        ssr_mode=False
    )