# SmartPlate Marine & Road

**YOLO-based vehicle license plate and ship identification-number detection** is the
planned project. Development begins with a deliberately small, runnable first stage:
an image-upload interface that runs an Ultralytics YOLO model, displays detections,
and downloads the annotated image and detection table.

## Stage 1 architecture and workflow

1. Streamlit accepts a JPG, JPEG, PNG, BMP, or WebP image.
2. The selected YOLO weights are loaded once and cached for the session.
3. OpenCV decodes the upload; Ultralytics runs inference at the chosen confidence.
4. The app displays the annotated image and each detected class, confidence, and box.
5. The annotated PNG and detection table can be downloaded.

The detection pipeline is in `src/detector.py`; the interface is in `app.py`.
Automated tests use mocked model and image-library boundaries and do not claim
real-world detection accuracy.

## Windows setup (CPU or GPU)

Use Python 3.11 or 3.12 in a VS Code terminal. Create and activate a virtual
environment from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the app:

```powershell
python -m streamlit run app.py
```

The first run with `yolo11n.pt` may download pretrained weights. Internet access
is therefore needed for that initial download. For CPU-only use, leave the default
Ultralytics device selection unchanged. For optional NVIDIA GPU acceleration,
install a PyTorch build compatible with your CUDA installation by following the
[official PyTorch install selector](https://pytorch.org/get-started/locally/), then
install the project requirements. Ultralytics selects an available supported
device automatically; CPU use remains available when CUDA is not installed.

## Test the first stage

The focused tests use Python's standard library:

```powershell
python -m unittest discover -s tests -v
```

Expected result: four passing tests for inference result formatting and invalid
image/confidence handling. These tests do not download weights or measure model
accuracy.

## Use a custom detector

Enter a local Ultralytics-compatible `.pt` weights path in the sidebar. The
pretrained `yolo11n.pt` model is a general COCO object detector; it is **not**
trained to detect license plates, vessel names, or IMO markings. A class such as
`car` is not a plate detection, and a detected text region is not confirmation of
an object's identity. Reliable plate and ship-marking detection requires separate,
appropriately labeled domain-specific training data.

## Current scope and next stages

This first stage does not implement OCR, plate/IMO validation, video, history,
database storage, dataset preparation, model training/evaluation, or identity
verification. They should be added in subsequent stages, with ship identification
kept distinct from vehicle plate detection. No accuracy metrics are reported until
they have been measured on an actual held-out test set.

Before publishing a trained model or dataset, check its license and usage terms.
Keep private images, datasets, trained weights, and generated YOLO runs out of
version control; the repository ignore rules cover common data and weight paths.

## Upload to your GitHub account

Create an empty repository on GitHub, then set its URL as `origin` and push:

```powershell
git remote add origin https://github.com/<your-username>/<your-repository>.git
git add .gitignore README.md app.py requirements.txt src tests
git commit -m "Add SmartPlate image detection starter"
git push -u origin HEAD
```
