# Building Footprint Detector

A lightweight aerial-image analysis app for detecting whether a building is present and, when present, segmenting the building footprint from the image.

This project combines a binary classification model and a segmentation model in a Streamlit interface. The app accepts an aerial image, predicts whether it contains a building, and overlays the predicted building mask on the input image for quick visual review.

## Overview

The workflow is:

1. Upload an aerial image.
2. A classifier decides whether a building is present.
3. If a building is detected, a segmentation model predicts the footprint mask.
4. The app displays the original image and the prediction overlay side by side.
5. Optional ground-truth masks can be compared using IoU and Dice metrics.

## Key Features

- Building presence classification using EfficientNet-B0
- Building footprint segmentation using DeepLabV3+ with a MobileNetV2 encoder
- Modern Streamlit interface for image upload and result viewing
- Downloadable prediction output image
- Optional IoU/Dice evaluation against a provided ground-truth mask
- Pretrained checkpoint files included in the repository

## Project Structure

```text
building_footprint_detector/
├── app.py                     # Streamlit web application
├── model.ipynb                # Training notebook for classifier and segmenter
├── requirements.txt           # Python dependencies
├── .gitignore                 # Local files excluded from version control
├── checkpoints/
│   ├── classifier_best.pt     # Best classifier checkpoint
│   └── segmenter_best.pt      # Best segmenter checkpoint
├── figures/
│   ├── architecture_diagram.png
│   ├── classifier_curves.png
│   ├── confusion_matrix.png
│   ├── metrics_summary.png
│   ├── sample_predictions.png
│   └── segmenter_curves.png
└── README.md                 # Project documentation
```

## Dependencies

The application uses:

- Python 3.x
- Streamlit
- PyTorch
- Torchvision
- segmentation-models-pytorch
- Albumentations
- Pillow
- NumPy

See `requirements.txt` for the exact package list.

## Installation

From the project root:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the App

After activating the environment:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

## How the App Works

### Classifier

The app loads a trained EfficientNet-B0 model from `checkpoints/classifier_best.pt` and predicts a probability that a building exists in the image.

### Segmenter

If the classifier decides that a building is present, the app loads a DeepLabV3+ segmentation model from `checkpoints/segmenter_best.pt` and generates a binary footprint mask.

### Visual Output

The output image is created by overlaying the predicted mask on the original image. A red-highlight style mask is applied to the image for a clear visual summary.

### Evaluation Mode

If a ground-truth mask is provided, the app computes:

- IoU (Intersection over Union)
- Dice coefficient

These are displayed as metrics alongside the prediction.

## Training Assets

The repository includes training results and visual summaries in the `figures/` folder, including:

- classifier learning curves
- segmentation learning curves
- confusion matrix
- metric summary
- sample predictions
- architecture diagram

The training workflow is captured in `model.ipynb`.

## Notes

- The app expects the checkpoint files to remain in `checkpoints/` with their current names.
- The figures in `figures/` are used in the training tab of the interface; if files are missing, the app displays a warning instead of failing.
- The project is designed for aerial image patches and works best with images aligned to the training distribution used in the notebook.

## License

This project does not include a dedicated license file. Please confirm the intended usage and distribution terms before sharing or reusing it outside the current workspace.
