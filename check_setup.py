"""Easy diagnostics for VISTA v4, including RapidOCR fallback."""
import sys
print('\nVISTA Setup Check  |  Python',sys.version.split()[0])
print('-'*55)
for pkg in ('streamlit','fitz','pytesseract','rapidocr','onnxruntime','cv2','zxingcpp','sklearn','PIL'):
    try:
        item=__import__(pkg)
        print(f'{pkg:14} OK', getattr(item,'__version__',''))
    except Exception as exc:
        print(f'{pkg:14} NOT INSTALLED:', str(exc)[:100])
try:
    from vista_core import BASE,MODEL_FILE,ocr_availability
    print('OCR status:',ocr_availability())
    print('FUNSD training forms:',len(list((BASE/'data'/'FUNSD'/'training_data'/'annotations').glob('*.json'))))
    print('FUNSD test forms:',len(list((BASE/'data'/'FUNSD'/'testing_data'/'annotations').glob('*.json'))))
    print('Saved model:', 'FOUND' if MODEL_FILE.exists() else 'MISSING')
except Exception as exc:
    print('Core check error:',exc)
print('-'*55)
print('Run: .venv\\Scripts\\python.exe -m streamlit run app.py')
