"""Offline image-decoding tests (actual QR and Code 128 demo images)."""
import sys
import unittest
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from code_scanner import decode_codes

BASE=Path(__file__).resolve().parents[1]

class BarcodeScannerTests(unittest.TestCase):
    def test_actual_demo_qr_image(self):
        codes,status=decode_codes([Image.open(BASE/'assets'/'demo_qr.png')])
        self.assertIn('VISTA-QR-DEMO-2026',[text for _,text in codes])

    def test_small_qr_in_photo_top_right(self):
        board=Image.new('RGB',(1000,640),(231,232,232))
        qr=Image.open(BASE/'assets'/'demo_qr.png').convert('RGB').resize((165,165))
        board.paste(qr,(755,44))
        codes,status=decode_codes([board],focus='Top right')
        self.assertIn('VISTA-QR-DEMO-2026',[text for _,text in codes])

    def test_actual_code128_image(self):
        # Decoder is ZXing on Windows; ZBar on dev machines if present.
        try:
            import zxingcpp
        except ImportError:
            try:
                from pyzbar.pyzbar import decode
            except (ImportError,OSError):
                self.skipTest('Native 1D barcode decoder not installed in this test environment')
        codes,status=decode_codes([Image.open(BASE/'assets'/'demo_barcode.png')])
        self.assertIn('VISTA-DEMO-2026',[text for _,text in codes])

if __name__=='__main__':
    unittest.main()
