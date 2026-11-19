# VISTA Studio v4 — Final-Year Document Intelligence Project

An easy-to-run Streamlit application for **OCR, document identification, summary, Q&A, QR codes, barcodes, translation, and FUNSD machine learning**. Includes all **199 real FUNSD scanned forms**, the trained field model, sample QR and Code128 barcode images, history, and dashboard charts.

## Windows / VS Code: Get started

1. Install **Python 3.11 64-bit** (https://www.python.org/downloads/).
2. **Stop your old VISTA app** using `Ctrl+C` in the previous terminal. Close the old browser tab.
3. **Extract the new ZIP**. In VS Code, open the **`VISTA_Studio_v4`** folder containing `app.py`.
4. Double-click **`START_VISTA.bat`** for automatic package installation and launch, or run these commands in a VS Code terminal:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.fileWatcherType none
```

5. Open **http://localhost:8501** in Chrome. Allow camera access when prompted.

> **Important:** Use the new folder / new `.venv`, not the Python environment of your old project. The saved ML model requires the listed scikit-learn version. First-time pip installation needs internet.

## Beginner-friendly workflow

**Tab 1 — Scan document:** Upload PDF/image or take a camera photo, click **Scan document**, then read the large **510-pixel-high recognized text panel**. View the original image/PDF page beside it. Below are summary, translation, questions, and details. The translated/original panels are **550 pixels high**. You can edit OCR errors and download the full text.

**Tab 2 — QR & barcode:** Choose **Upload**, **Camera**, **Try demo QR**, or **Try demo barcode**. Press **Read QR / barcode**. If your QR is small or off-center, choose **Top right**, **Center**, etc. from **Where is the code in the image?** and try again. The camera photo must include **all four QR corners** and a little white margin; a clipped, blurry, or reflecting code may not be decodable. The two demo images under `assets/` help verify decoding without camera setup.

**Tab 3 — Dashboard & ML:** Measured FUNSD field-label accuracy, training/testing dataset counts, per-label F1 chart, history category chart, confusion matrix, and setup diagnostics.

**Tab 4 — History:** Save, review, export, or delete your own locally saved scans.

## Barcode/QR fix in v4

- We added **`zxing-cpp==3.1.1`** to `requirements.txt`. Its Windows/Python 3.11 package includes a QR and **1D barcode** scanner without a separate ZBar DLL install. It supports common formats including QR, Code128, EAN, and Code39.
- Offline fallback: optional `pyzbar` (if installed) and OpenCV QR reader. `code_scanner.py` checks the full image plus center and overlapping corners, and tries grayscale/contrast/border processing.
- **If demo QR works but camera QR fails**, the problem is likely the photo: show the **entire** QR without clipping and retake with sharper focus/less glare.
- **If demo barcode fails**, run `.venv\Scripts\python.exe -m pip show zxing-cpp` and ensure the package installed successfully. Re-run the package installation command above if it is missing.

## OCR, translation and ML notes

- Printed English OCR uses Tesseract if installed or a Python RapidOCR fallback; RapidOCR may download model files on first use. Hindi/Japanese handwriting and non-English printed text may need separate language packs/EasyOCR.
- Translation uses an online, **rate-limited free service** and translates up to 1,400 characters per request. It is not guaranteed offline.
- The **document type** feature is rule-based classification of recognized words (Invoice, Government ID, Academic, etc.), not a trained image classifier or ID-verification service.
- The trained **FUNSD ML model** classifies document **fields** (Question / Answer / Header / Other); its measured held-out accuracy is **68.15%** on annotated fields. It is **not** document-type or OCR accuracy. See `models/evaluation.json`.
- Do not store sensitive ID numbers or private medical details in this classroom demo: local SQLite history is not encrypted.

## Developer tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe train_model.py
```

Some tests require actual OCR libraries or barcode decoder availability. Webcam/browser end-to-end testing must be done on your own Windows laptop.
