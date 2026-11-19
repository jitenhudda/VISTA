"""VISTA's small, testable document utilities. No Streamlit state is stored here."""
from __future__ import annotations

import datetime as dt
import io
import re
import shutil
import sqlite3
from collections import Counter
from functools import lru_cache
from pathlib import Path

import fitz
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import pytesseract

BASE = Path(__file__).resolve().parent
DB = BASE / 'vista_history.db'
MODEL_FILE = BASE / 'models' / 'funsd_classifier.joblib'
OCR_LANGS = {'English': 'eng', 'Hindi': 'hin', 'Spanish': 'spa', 'Japanese': 'jpn'}
TRANSLATE_LANGS = {'English': 'en', 'Hindi': 'hi', 'Spanish': 'es', 'Japanese': 'ja'}


def configure_tesseract() -> bool:
    """Find Tesseract in PATH or standard Windows install folders."""
    found = shutil.which('tesseract')
    if not found:
        for candidate in (
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ):
            if Path(candidate).is_file():
                found = candidate
                break
    if found:
        pytesseract.pytesseract.tesseract_cmd = str(found)
    return bool(found)


def prepare_image(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert('RGB')
    # Improve photos taken of paper held in front of a laptop webcam.
    if max(image.size) < 1700:
        factor = min(2, 1700 / max(image.size))
        image = image.resize((round(image.width * factor), round(image.height * factor)), Image.Resampling.LANCZOS)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.22)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=125, threshold=3))
    return image


@lru_cache(maxsize=1)
def _rapid_engine():
    """Local ONNX OCR engine. Runs without a separate Tesseract.exe install.

    RapidOCR may download model weights on first use, depending on version.
    """
    from rapidocr import RapidOCR
    return RapidOCR()


def ocr_availability() -> str:
    """Report available engine without constructing/downloading a large model."""
    if configure_tesseract():
        return 'Tesseract ready'
    try:
        import rapidocr  # noqa: F401
        import onnxruntime  # noqa: F401
        return 'RapidOCR ready (first scan may download OCR models)'
    except ImportError:
        return 'OCR missing: install requirements.txt (RapidOCR) or Tesseract OCR'


def _rapid_text(image: Image.Image) -> str:
    try:
        result = _rapid_engine()(np.asarray(image))
        lines = result.txts or []
        return '\n'.join(str(x).strip() for x in lines if x and str(x).strip())
    except Exception as exc:
        raise RuntimeError(
            'RapidOCR could not start. Check your internet connection on the first scan '
            'to download OCR models, or install Tesseract OCR. '
            f'Details: {str(exc)[:170]}'
        ) from exc


def _quality(text: str) -> int:
    """A conservative legibility heuristic, for sideways camera-paper fallback."""
    words = re.findall(r'[A-Za-z]{4,}', text)
    return sum(min(20, len(word) ** 2) for word in words)


def _rotate_if_unreadable(image: Image.Image, initial: str, reader) -> str:
    """Try sideways orientations only when original has very few readable words."""
    if _quality(initial) >= 65:
        return initial
    candidates = [initial]
    for angle in (90, 270):
        try:
            candidates.append(reader(image.rotate(angle, expand=True)))
        except Exception:
            pass
    return max(candidates, key=_quality)


def ocr_image(image: Image.Image, language: str = 'English', handwriting: bool = False) -> str:
    image = prepare_image(image)
    if handwriting:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError('Handwriting recognition needs optional EasyOCR: .venv\\Scripts\\python.exe -m pip install easyocr') from exc
        lang = {'English': 'en', 'Hindi': 'hi', 'Spanish': 'es', 'Japanese': 'ja'}[language]
        reader = easyocr.Reader([lang] if lang == 'en' else [lang, 'en'], gpu=False)
        return '\n'.join(reader.readtext(np.asarray(image), detail=0, paragraph=True)).strip()

    code = OCR_LANGS.get(language, 'eng')
    if configure_tesseract():
        available = pytesseract.get_languages(config='')
        if code in available:
            result = pytesseract.image_to_string(image, lang=code, config='--oem 3 --psm 3').strip()
            if len(re.sub(r'\W+', '', result)) < 12:
                alternative = pytesseract.image_to_string(image, lang=code, config='--oem 3 --psm 6').strip()
                if len(re.sub(r'\W+', '', alternative)) > len(re.sub(r'\W+', '', result)):
                    result = alternative
            if language == 'English':
                result = _rotate_if_unreadable(
                    image, result,
                    lambda photo: pytesseract.image_to_string(photo, lang=code, config='--oem 3 --psm 6').strip(),
                )
            return result
    # No separate EXE required. RapidOCR works well for common printed English.
    if language in ('English', 'Spanish'):
        try:
            primary = _rapid_text(image)
            return _rotate_if_unreadable(image, primary, _rapid_text)
        except ImportError as exc:
            raise RuntimeError('OCR not installed. Run: .venv\\Scripts\\python.exe -m pip install -r requirements.txt') from exc
    raise RuntimeError(f'{language} OCR needs the matching Tesseract language pack or optional EasyOCR. See README.md.')


def read_document(data: bytes, filename: str, language: str = 'English', handwriting: bool = False, max_pages: int = 5) -> dict:
    """Fast PDF selectable-text extraction; render pages only when OCR is required.

    This avoids keeping several high-resolution PDF page images in memory,
    which can make Streamlit unresponsive on modest Windows laptops.
    """
    if len(data) > 40 * 1024 * 1024:
        raise RuntimeError('File too large. Choose an image/PDF smaller than 40 MB.')
    suffix = Path(filename).suffix.lower()
    images: list[Image.Image] = []
    sections: list[str] = []
    page_count = 0
    processed = 0
    if suffix == '.pdf':
        try:
            document = fitz.open(stream=data, filetype='pdf')
        except Exception as exc:
            raise RuntimeError('Unable to open PDF. It may be corrupted or password-protected.') from exc
        try:
            if document.needs_pass:
                raise RuntimeError('Password-protected PDF. Please unlock it first.')
            page_count = len(document)
            for index in range(min(page_count, max_pages)):
                page = document[index]
                processed += 1
                extracted = page.get_text('text', sort=True).strip()
                if len(re.sub(r'\s+', '', extracted)) < 25:
                    # 1.65x is adequate for most scans without excessive memory use.
                    pix = page.get_pixmap(matrix=fitz.Matrix(1.65, 1.65), alpha=False)
                    image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
                    extracted = ocr_image(image, language, handwriting)
                if extracted:
                    sections.append(extracted)
        finally:
            document.close()
    elif suffix in {'.jpg', '.jpeg', '.png', '.webp'}:
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
            image = ImageOps.exif_transpose(image).convert('RGB')
        except Exception as exc:
            raise RuntimeError('Could not read image. Try a JPG or PNG.') from exc
        page_count = processed = 1
        extracted = ocr_image(image, language, handwriting)
        if extracted:
            sections.append(extracted)
    else:
        raise RuntimeError('Unsupported file. Use PDF, JPG, PNG or WEBP.')
    return {'text': '\n\n'.join(sections), 'pages': page_count, 'processed_pages': processed, 'images': images}


def decode_codes(images: list[Image.Image], focus: str = 'Automatic (recommended)') -> tuple[list[tuple[str,str]], str]:
    """Stable public wrapper for the enhanced ZXing/OpenCV scanner."""
    from code_scanner import decode_codes as enhanced_decode
    return enhanced_decode(images, focus=focus)


def detect_language(text: str) -> str:
    if not text.strip():
        return 'Unknown'
    try:
        from langdetect import detect
        return {'en': 'English', 'hi': 'Hindi', 'es': 'Spanish', 'ja': 'Japanese'}.get(detect(text[:4000]), 'Other language')
    except Exception:
        return 'Unknown'


def split_sentences(text: str) -> list[str]:
    raw = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [re.sub(r'\s+', ' ', s).strip() for s in raw if len(s.strip()) >= 12]


def summarize_text(text: str, maximum: int = 4) -> str:
    """Simple extractive summary, transparent and fully offline."""
    lines = split_sentences(text)
    if not lines:
        return 'No readable text to summarize. Try a clearer photo.'
    if len(lines) <= maximum:
        return '\n\n'.join(lines)
    stop = {'the','this','that','and','for','with','from','have','was','were','are','your','will','you','their','they','into','there','what','which','does','of','in','on','to','by','it','as','an'}
    words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    counts = Counter(w for w in words if w not in stop)
    scores = [(i, sum(counts[w] for w in re.findall(r'[a-zA-Z]{3,}', line.lower())) / max(1, len(line.split())**0.5)) for i, line in enumerate(lines)]
    selected = {i for i, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:maximum]}
    return '\n\n'.join(lines[i] for i in range(len(lines)) if i in selected)


def answer_question(text: str, question: str) -> str:
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    if not text.strip():
        return 'Please scan a document first.'
    if not question.strip():
        return 'Type a question about the document.'
    q = set(re.findall(r'[\w]{3,}', question.lower())) - {'the','what','is','was','are','were','who','when','where','how','which','about','please','show','tell','document','this','that','from','give','me'}
    if not q:
        return 'Ask using a keyword from your document, such as date, amount or course.'
    candidates = lines if len(lines) > 1 else split_sentences(text)
    scores = []
    for i, line in enumerate(candidates):
        tokens = set(re.findall(r'[\w]{3,}', line.lower()))
        match = len(q & tokens)
        # A matching label may be followed by its answer on the next line.
        if match:
            snippet = line
            if line.endswith(':') and i + 1 < len(candidates):
                snippet += ' ' + candidates[i + 1]
            scores.append((match, -i, snippet))
    if not scores:
        return 'I could not find a reliable matching line. Try a question with words found in the scanned text.'
    scores.sort(reverse=True)
    return '\n\n'.join(item[2] for item in scores[:2])


# Document type is identified with readable words from OCR, not with an
# image-trained model. This is intentionally transparent for academic evaluation.
# Each regex contributes evidence to a type. Stronger types have priority for ties.
DOCUMENT_RULES = {
    'Government ID': [
        (r'\b(?:aadhaar|aadhar|uidai|unique identification|voter id|election commission|identity card)\b', 4),
        (r'\b(?:pan card|permanent account number|driving licen[cs]e|driver.s licen[cs]e)\b', 4),
        (r'\b(?:government of india|govt\.? of india|identity proof)\b', 3),
        (r'\b(?:date of birth|dob)\b', 1),
    ],
    'Invoice': [
        (r'\b(?:tax invoice|invoice(?:\s*(?:number|no\.?|#))?)\b', 5),
        (r'\b(?:bill to|billed to|ship to|invoice date)\b', 2),
        (r'\b(?:gstin|hsn|cgst|sgst|igst)\b', 2),
        (r'\b(?:subtotal|grand total|amount due|total payable)\b', 2),
        (r'\b(?:unit price|quantity|qty)\b', 1),
    ],
    'Receipt': [
        (r'\b(?:receipt|payment received|payment confirmation)\b', 5),
        (r'\b(?:change due|cash paid|transaction id|payment method)\b', 2),
    ],
    'Passport / Visa': [
        (r'\b(?:passport|visa(?: number)?|immigration|republic of india)\b', 4),
        (r'\b(?:nationality|place of birth|date of expiry)\b', 1),
    ],
    'Travel Ticket': [
        (r'\b(?:boarding pass|flight no\.?|train ticket|e-ticket|pnr)\b', 5),
        (r'\b(?:departure|arrival|seat number|gate number|booking reference)\b', 2),
    ],
    'Medical / Prescription': [
        (r'\b(?:prescription|diagnosis|patient id|medical report|laboratory report)\b', 5),
        (r'\b(?:patient name|doctor|hospital|clinic|blood group|dosage)\b', 2),
        (r'\b(?:medication|medicine|symptoms|treatment)\b', 1),
    ],
    'Academic Document': [
        (r'\b(?:university|college|school|semester|marksheet|transcript)\b', 4),
        (r'\b(?:course title|course code|student name|roll number|assignment|examination)\b', 3),
        (r'\b(?:subject|grade|class|practical file|submitted to)\b', 1),
    ],
    'Product Packaging': [
        (r'\b(?:ingredients|nutrition facts|net weight|manufactured by|marketed by|mrp|best before)\b', 4),
        (r'\b(?:filter cigarette(?:s)?|cigarette(?:s)?|tobacco product|batch no|packed on|warning:?)\b', 3),
        (r'\b(?:variance|menthol|pack of|retail price)\b', 1),
    ],
    'Business Document': [
        (r'\b(?:purchase order|quotation|estimate|balance sheet|business proposal)\b', 4),
        (r'\b(?:company|financial|budget|payment)\b', 1),
    ],
    'Form': [
        (r'\b(?:application form|registration form|form no\.?|please fill|signature of applicant)\b', 4),
        (r'\b(?:name|address|signature|phone number)\b', 1),
    ],
}
TYPE_TO_CATEGORY = {
    'Government ID': 'Government / Identity', 'Invoice': 'Business', 'Receipt': 'Business',
    'Passport / Visa': 'Travel', 'Travel Ticket': 'Travel',
    'Medical / Prescription': 'Healthcare', 'Academic Document': 'Education',
    'Product Packaging': 'Product / Packaging', 'Business Document': 'Business',
    'Form': 'General', 'General Document': 'General',
}


def detect_document_type(text: str) -> dict:
    """OCR-text-driven category and evidence, NOT visual ML confidence."""
    if not text or not text.strip():
        return {'type': 'Unrecognized', 'category': 'General', 'strength': 'Unknown',
                'evidence': [], 'method': 'OCR keyword patterns'}
    scored = []
    for kind, expressions in DOCUMENT_RULES.items():
        total, cues = 0, []
        for pattern, weight in expressions:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                total += weight
                cues.append(match.group(0)[:55])
        scored.append((total, kind, cues))
    # priority is the insertion order; sorting only by score keeps ties stable
    winner = max(scored, key=lambda row: row[0])
    if winner[0] < 3:
        return {'type':'General Document', 'category':'General', 'strength':'Low',
                'evidence': [], 'method':'OCR keyword patterns'}
    strength = 'High' if winner[0] >= 7 and len(winner[2]) >= 2 else ('Medium' if winner[0] >= 4 else 'Low')
    return {'type':winner[1], 'category':TYPE_TO_CATEGORY[winner[1]],
            'strength':strength, 'evidence':winner[2], 'method':'OCR keyword patterns'}


def document_category(text: str) -> str:
    return detect_document_type(text)['category']


def extract_details(text: str, category: str) -> dict[str,str]:
    patterns = {
        'Emails': r'\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b',
        'Dates found': r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b',
    }
    if category == 'Business':
        patterns.update({
            'Invoice number': r'\binvoice\s*(?:no\.?|number|#)\s*[:#\-]?\s*([A-Za-z0-9/-]+)',
            'Total amount': r'\b(?:grand\s+total|total\s+amount|amount\s+due)\s*[:₹$\s]*([\d,]+(?:\.\d{1,2})?)',
        })
    if category == 'Education':
        patterns.update({'Course': r'\bcourse\s*(?:title|name)?\s*[:\-]\s*([^\n]+)', 'Student': r'\bstudent\s*(?:name)?\s*[:\-]\s*([^\n]+)'})
    if category == 'Healthcare':
        patterns.update({'Patient': r'\bpatient\s*(?:name)?\s*[:\-]\s*([^\n]+)', 'Doctor': r'\bdoctor\s*(?:name)?\s*[:\-]\s*([^\n]+)'})
    if category == 'Travel':
        patterns.update({'Flight': r'\bflight\s*(?:no\.?|number)?\s*[:\-]\s*([A-Za-z0-9-]+)'})
    details = {}
    for label, pattern in patterns.items():
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            details[label] = ', '.join(dict.fromkeys(str(item).strip()[:80] for item in matches[:6]))
    return details


def explain_document(text: str) -> str:
    """Beginner-friendly rule-based overview, clearly not generative AI."""
    if not text.strip():
        return 'No text was recognized. Scan a clearer page to see an explanation.'
    category = document_category(text)
    names = {
        'Healthcare': 'healthcare or medical',
        'Business': 'business or payment',
        'Education': 'education or academic',
        'Travel': 'travel',
        'Government / Identity': 'identity or government',
        'General': 'general',
        'Product / Packaging': 'product packaging',
    }
    intro = f'This appears to be a document related to {names.get(category, "general information")}.'
    fields = extract_details(text, category)
    if fields:
        pairs = [f'{key.lower()}: {value[:100]}' for key, value in list(fields.items())[:3]]
        return intro + ' Important information found: ' + '; '.join(pairs) + '.'
    key_lines = summarize_text(text, 2).replace('\n\n', ' / ')
    return intro + ' Main points: ' + key_lines[:360]


def _read_date(value: str):
    try:
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            return dt.date.fromisoformat(value)
        # DD-MM-YYYY is unambiguous only when specified; validate conditional.
        parts = re.split(r'[/\-]', value)
        if len(parts) == 3:
            day, month, year = map(int, parts)
            year += 2000 if year < 100 else 0
            return dt.date(year, month, day)
    except ValueError:
        return None
    return None


def validate_document(text: str, category: str) -> list[str]:
    messages: list[str] = []
    if category in {'Travel','Government / Identity'}:
        found = re.findall(r'\b(?:expiry|expires|valid\s+until|expiration)\s*(?:date)?\s*[:\-]?\s*(\d{4}-\d{2}-\d{2}|\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})', text, re.I)
        if not found:
            messages.append('Expiry date not confidently found; please check the document yourself.')
        for value in found:
            parsed = _read_date(value)
            if parsed:
                messages.append(f'Expiry {value}: '+('Expired' if parsed < dt.date.today() else 'Not expired as of today')+' (assuming DD/MM/YYYY where applicable)')
            else:
                messages.append(f'Expiry {value}: date format could not be verified.')
    if category == 'Business':
        patterns = {
            'Invoice number': r'\binvoice\s*(?:number|no\.?|#)\s*[:\-#]?\s*([A-Za-z0-9/-]+)',
            'Total amount': r'\b(?:grand\s+total|total\s+amount|amount\s+due)\s*[:₹$\s]*([\d,]+(?:\.\d{1,2})?)',
        }
        for label, pattern in patterns.items():
            found = re.search(pattern, text, re.I)
            messages.append(f'{label}: {found.group(1)}' if found else f'{label}: not found; please verify manually.')
        messages.append('Values are extracted only; financial correctness has not been confirmed.')
    if not messages:
        messages.append('No automatic expiry or invoice check applies to this document type.')
    return messages


def translate_text(text: str, language: str) -> str:
    """Limited-length online translation with explicit connection timeout.

    Uses Google's public translate endpoint; access is subject to rate limits.
    A service failure raises a readable exception instead of silently hanging.
    """
    if not text.strip():
        return ''
    if language not in TRANSLATE_LANGS:
        raise ValueError('Unsupported translation language')
    from urllib import parse
    import requests
    # Short passages are safer for an academic demo and avoid hitting rate limits.
    source = 'auto'
    target = TRANSLATE_LANGS[language]
    try:
        response = requests.get(
            'https://translate.googleapis.com/translate_a/single',
            params={'client':'gtx','sl':source,'tl':target,'dt':'t','q':text[:1400]},
            timeout=(4, 9),
            headers={'User-Agent':'Mozilla/5.0'},
        )
        response.raise_for_status()
        answer = response.json()
        result = ''.join(part[0] or '' for part in answer[0] if part and part[0])
        if not result.strip():
            raise ValueError('Translation service returned no text')
        return result
    except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as exc:
        raise RuntimeError('Online translation is unavailable or rate-limited. Please retry later.') from exc


def speech_mp3(text: str, language: str = 'English') -> bytes:
    from gtts import gTTS
    buffer = io.BytesIO()
    # Bound gTTS HTTP calls; do not stall the whole UI waiting indefinitely.
    gTTS(text=text[:1200], lang=TRANSLATE_LANGS.get(language, 'en'), timeout=(4, 8)).write_to_fp(buffer)
    return buffer.getvalue()


def init_db():
    conn = sqlite3.connect(DB)
    conn.execute('''CREATE TABLE IF NOT EXISTS scans(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT NOT NULL,
      filename TEXT NOT NULL,
      category TEXT,
      text TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )''')
    return conn


def save_history(username: str, filename: str, category: str, text: str) -> None:
    with init_db() as conn:
        conn.execute('INSERT INTO scans(username,filename,category,text) VALUES(?,?,?,?)', (username, filename, category, text))


def list_history(username: str):
    with init_db() as conn:
        return conn.execute('SELECT id,filename,category,created_at,text FROM scans WHERE username=? ORDER BY id DESC LIMIT 100', (username,)).fetchall()


def delete_history(username: str, row_id: int) -> None:
    with init_db() as conn:
        conn.execute('DELETE FROM scans WHERE username=? AND id=?', (username, row_id))
