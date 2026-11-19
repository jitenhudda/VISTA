"""Run from project root: python -m unittest discover -s tests -v"""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import fitz
import joblib
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vista_core as v


class DocumentTests(unittest.TestCase):
    def test_summary_and_questions(self):
        text = ('Course Title: Deep Learning\nCourse Code: CSE 407\n'
                'Student Name: Alex Student\nExam date: 23/11/2026')
        self.assertIn('Deep Learning', v.summarize_text(text))
        self.assertIn('Deep Learning', v.answer_question(text, 'What is the course title?'))
        self.assertEqual(v.document_category(text), 'Education')
        self.assertIn('Course', v.extract_details(text, 'Education'))

    def test_invoice_validation(self):
        text = 'Invoice number: AB-56\nTotal Amount: 234.00\nGST: 6'
        self.assertEqual(v.document_category(text), 'Business')
        checks = v.validate_document(text, 'Business')
        self.assertTrue(any('234.00' in note for note in checks))

    def test_recognize_document_types_from_ocr_text(self):
        examples = {
            'Invoice': 'TAX INVOICE Invoice No AB-2 GSTIN Amount Due 400',
            'Government ID': 'Government of India UIDAI Aadhaar Date of Birth: 1/1/2001',
            'Academic Document': 'University Course Title - Deep Learning Course Code: CSE 407',
            'Product Packaging': 'VARIANCE MENTHOL FILTER CIGARETTE',
            'Travel Ticket': 'BOARDING PASS Flight No AL123 Departure Time',
        }
        for expected, sample in examples.items():
            with self.subTest(expected=expected):
                self.assertEqual(v.detect_document_type(sample)['type'], expected)

    def test_ocr_without_tesseract_uses_rapidocr_fallback(self):
        from unittest.mock import patch
        with patch.object(v, 'configure_tesseract', return_value=False), patch.object(v, '_rapid_text', return_value='Invoice Number INV-123 GSTIN'):
            result = v.ocr_image(Image.new('RGB', (500, 260), color='white'))
            self.assertIn('Invoice Number', result)

    def test_pdf_text_extraction(self):
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((70, 80), 'Course Title: Deep Learning')
        document = v.read_document(pdf.tobytes(), 'learning.pdf')
        pdf.close()
        self.assertEqual(document['pages'], 1)
        self.assertIn('Deep Learning', document['text'])

    def test_real_funsd_image_ocr(self):
        images = sorted((v.BASE / 'data' / 'FUNSD' / 'testing_data' / 'images').glob('*.png'))
        self.assertEqual(len(images), 50)
        if not v.configure_tesseract():
            self.skipTest('Tesseract unavailable in this test environment')
        extracted = v.read_document(images[0].read_bytes(), images[0].name)
        self.assertGreater(len(extracted['text']), 100)
        self.assertEqual(extracted['pages'], 1)

    def test_qr_on_captured_photo(self):
        try:
            import qrcode
        except ImportError:
            self.skipTest('qrcode development test dependency not installed')
        image = qrcode.make('VISTA-CAMERA-QR-123').convert('RGB').resize((600,600))
        found, _ = v.decode_codes([image])
        self.assertTrue(any('VISTA-CAMERA-QR-123' == item[1] for item in found))

    def test_pretrained_model(self):
        model = joblib.load(v.MODEL_FILE)
        result = model.predict_proba(['Course Title: Deep Learning', 'Total Amount: 300'])
        self.assertEqual(result.shape, (2,4))
        eval_path = v.BASE / 'models' / 'evaluation.json'
        metrics = json.loads(eval_path.read_text(encoding='utf-8'))
        self.assertEqual(metrics['training_documents'], 149)
        self.assertEqual(metrics['test_documents'], 50)
        self.assertGreater(metrics['accuracy'], 0.6)


if __name__ == '__main__':
    unittest.main()
