# Deepfake Image Detection using CNN

A convolutional neural network that classifies face images as **real** or **AI-generated**, wrapped in a FastAPI backend that accepts an uploaded image and returns a prediction.

The model is a custom CNN trained from scratch (no transfer learning) on real faces from FFHQ and synthetic faces from ThisPersonDoesNotExist.

## Results

Evaluated on a held-out validation split of 1,200 images (600 real, 600 fake, roughly), with "fake" as the positive class.

| Metric | Score |
|--------|-------|
| Accuracy | 99.0% (1,188 / 1,200 correct) |
| Precision | 98.9% |
| Recall | 99.2% |

|  | Predicted real | Predicted fake |
|--|----------------|----------------|
| **Actually real** | 586 | 7 |
| **Actually fake** | 5 | 602 |

Training ran for 100 epochs. The best checkpoint (by validation accuracy, 99.0%) was saved and used for these numbers.

**Read these numbers with some caution.** Both classes come from one real dataset and one generator, and the same validation split was used to pick the best checkpoint, so there is no fully untouched test set. Real-world accuracy on other generators or compressed images will likely be lower (see Limitations).

## Dataset

| Class | Source | Images used |
|-------|--------|-------------|
| Real | FFHQ face thumbnails (128×128) | 3,000 |
| Fake | ThisPersonDoesNotExist (StyleGAN faces) | 3,000 |

- Images are loaded at 128×128 and scaled to the 0–1 range.
- Random 80/20 split: 4,800 training and 1,200 validation images.
- The datasets are not stored in this repo. Both are available on Kaggle (the notebook reads them from `ffhq-face-data-set` and `person-face-dataset-thispersondoesnotexist`).

## Model

A 10-layer CNN with about 653k parameters.

- **Convolutional part:** three 5×5 conv layers with 64 filters, followed by three 3×3 conv layers (32, 16, 16 filters) with dilation rate 2 to widen the receptive field. Every conv layer is followed by ReLU and max pooling.
- **Classifier head:** global average pooling, then dense layers of 400, 512 and 400 units, each with ReLU and 50% dropout, and a 2-unit sigmoid output.
- **Training setup:** Adam optimizer (learning rate 1e-5), binary cross-entropy loss, batch size 32, 100 epochs, with a checkpoint saved whenever validation accuracy improved.

## How the API works

`app.py` exposes two routes:

| Route | Method | Input | Purpose |
|-------|--------|-------|---------|
| `/` | GET | none | Health check |
| `/predictImage` | POST | form field `image` | Classify one image |

The prediction route returns JSON like `{"result": 1}`:

| Value | Meaning |
|-------|---------|
| `0` | Real |
| `1` | Deepfake |
| `-1` | No face detected (images) |

Before classification, `features.py` finds a face with OpenCV's Haar cascade detector, crops it with a margin, resizes it and scales the pixels.

## Run it locally

Requires Python 3.11 (see `runtime.txt`).

```bash
git clone https://github.com/dhrnshwn-13/Deep-Fake-Image-Detection-using-CNN.git
cd Deep-Fake-Image-Detection-using-CNN

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install python-multipart     # needed by FastAPI for file uploads

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

`features.py` loads the trained model from `./models/deepfake2.keras`. **[FILL IN: that file isn't in the repo. Say where to download it, or how to generate it.]**

Test it:

```bash
curl -X POST -F "image=@face.jpg" http://localhost:8000/predictImage
```

**[CONFIRM: `training.ipynb` trains on 128×128 inputs, but `features.py` resizes faces to 224×224 before prediction. State which input size the deployed model expects, and whether `deepfake2.keras` is the model from the notebook.]**

## Limitations

- Trained on one real dataset (FFHQ) and one generator (StyleGAN). It has not been tested on diffusion-model images, face swaps or heavily compressed photos, so don't expect 99% outside this setup.
- Face detection uses a Haar cascade, which misses faces that are small, rotated or partly hidden.
- No data augmentation was applied during training.

## What I'd improve next

- Hold out a separate test set that is never used for checkpoint selection
- Test on fakes from other generators to measure how well the model generalizes
- Add Grad-CAM heatmaps to show which parts of the face drive a prediction

## Tech stack

Python, TensorFlow/Keras, OpenCV, FastAPI, Uvicorn

## Author

**Dharun** · [GitHub](https://github.com/dhrnshwn-13)
