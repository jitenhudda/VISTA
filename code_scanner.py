"""Resilient offline QR/1D barcode reading for snapshots, paper and screens.

Primary engine: zxing-cpp (bundled wheels for Python 3.11 on Windows).
Fallback: optional pyzbar if ZBar is available; OpenCV for QR.
Do not invent decoded data; only return actual decoder outputs.
"""
from __future__ import annotations

from typing import Iterable
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

FOCUS_OPTIONS = ('Automatic (recommended)', 'Whole picture', 'Center',
                 'Top left', 'Top right', 'Bottom left', 'Bottom right')


def _resize_max(image: Image.Image, limit: int = 1900) -> Image.Image:
    if max(image.size) <= limit:
        return image
    k = limit / max(image.size)
    return image.resize((max(1, int(image.width*k)), max(1, int(image.height*k))), Image.Resampling.LANCZOS)


def _regions(image: Image.Image, focus: str) -> list[tuple[str, Image.Image]]:
    """Automatic scan checks the whole picture and overlapping focus zones."""
    img = _resize_max(ImageOps.exif_transpose(image).convert('RGB'))
    w,h = img.size
    def clip(name,box):
        x0,y0,x1,y1=box
        return (name, img.crop((int(x0*w),int(y0*h),int(x1*w),int(y1*h))))
    regions = {'Center':(.18,.18,.82,.82),
               'Top left':(0,0,.68,.68), 'Top right':(.32,0,1,.68),
               'Bottom left':(0,.32,.68,1), 'Bottom right':(.32,.32,1,1)}
    all_parts=[('Whole picture',img)]
    if focus == 'Whole picture':
        return all_parts
    if focus in regions:
        return [clip(focus,regions[focus]),('Whole picture',img)]
    return all_parts+[clip(label,region) for label,region in regions.items()]


def _preprocess(image: Image.Image):
    """Limited enhancement, to avoid slow repeated high-resolution decoding."""
    rgb=np.asarray(image.convert('RGB'))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    enhanced=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
    # Reflect-padding restores some margin around codes near crop edges.
    padded=cv2.copyMakeBorder(gray,22,22,22,22,cv2.BORDER_CONSTANT,value=255)
    return [('color',rgb),('grayscale',gray),('contrast',enhanced),('border',padded)]


def _zxing_decode(arr):
    import zxingcpp
    return [(str(item.format).replace('BarcodeFormat.', ''),str(item.text))
            for item in zxingcpp.read_barcodes(arr) if str(item.text).strip()]


def _opencv_decode(arr, detector):
    output=[]
    try:
        value,_,_=detector.detectAndDecode(arr)
        if value:
            output.append(('QR',value))
    except (cv2.error, ValueError):
        pass
    if not output:
        try:
            ok,values,_,_=detector.detectAndDecodeMulti(arr)
            if ok:
                output.extend(('QR',val) for val in values if val)
        except (cv2.error, ValueError):
            pass
    return output


def decode_codes(images: Iterable[Image.Image], focus: str = 'Automatic (recommended)') -> tuple[list[tuple[str,str]], str]:
    """Decode QR/EAN/Code128/Code39 etc; return code list plus honest engine status."""
    if focus not in FOCUS_OPTIONS:
        focus=FOCUS_OPTIONS[0]
    found=[]
    try:
        import zxingcpp  # noqa: F401
        has_zxing=True
    except ImportError:
        has_zxing=False
    detector=cv2.QRCodeDetector()
    try:
        from pyzbar.pyzbar import decode as zbar_decode
        has_zbar=True
    except (ImportError, OSError):
        has_zbar=False
    for img in images:
        if img is None:
            continue
        regions=_regions(img,focus)
        for location,region_img in regions:
            variants=_preprocess(region_img)
            # ZXing reads most common QR + 1D barcodes on Windows without ZBar DLLs.
            if has_zxing:
                for kind,arr in variants[:3]:
                    try:
                        for fmt,value in _zxing_decode(arr):
                            if (fmt,value) not in found:
                                found.append((fmt,value))
                    except (ValueError,RuntimeError,TypeError):
                        pass
                    if found:
                        break
            if not found and has_zbar:
                for kind,arr in variants[:3]:
                    try:
                        for item in zbar_decode(Image.fromarray(arr)):
                            fmt,value=item.type,item.data.decode('utf-8',errors='replace')
                            if value and (fmt,value) not in found:
                                found.append((fmt,value))
                    except (OSError,ValueError,TypeError):
                        pass
                    if found:
                        break
            if not found:
                for kind,arr in variants:
                    for fmt,value in _opencv_decode(arr,detector):
                        if (fmt,value) not in found:
                            found.append((fmt,value))
                    if found:
                        break
            if found:
                break
    if has_zxing:
        status='ZXing scanner available (QR, EAN, Code128 and other common barcode formats).'
    elif has_zbar:
        status='Using ZBar fallback for QR and common 1D barcodes. Install ZXing for broader support.'
    else:
        status='Only OpenCV QR fallback available. Run pip install zxing-cpp to enable 1D barcode scanning.'
    return found,status
