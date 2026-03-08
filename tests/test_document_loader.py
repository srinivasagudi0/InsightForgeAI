from __future__ import annotations

import shutil
from io import BytesIO

import pytest
from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from document_loader import load_text_input, load_uploaded_document


def test_load_text_input_metadata() -> None:
    document = load_text_input("notes.txt", "Alpha beta gamma")
    assert document.name == "notes.txt"
    assert document.source_type == "Text"
    assert document.word_count == 3
    assert document.used_ocr is False


def test_load_uploaded_pdf_extracts_text_without_ocr() -> None:
    pdf_bytes = build_text_pdf("Research methods and findings.")
    document = load_uploaded_document("paper.pdf", pdf_bytes)

    assert document.source_type == "PDF"
    assert document.page_count == 1
    assert "Research methods and findings." in document.content
    assert document.used_ocr is False


@pytest.mark.skipif(
    shutil.which("tesseract") is None,
    reason="tesseract is not installed",
)
def test_load_uploaded_scanned_pdf_uses_ocr() -> None:
    pdf_bytes = build_image_pdf("SCANNED PDF OCR")
    document = load_uploaded_document("scanned.pdf", pdf_bytes)

    assert document.source_type == "PDF"
    assert document.used_ocr is True
    assert "SCANNED PDF OCR" in document.content


def build_text_pdf(text: str) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=(500, 200))
    pdf.drawString(40, 120, text)
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def build_image_pdf(text: str) -> bytes:
    image = Image.new("RGB", (500, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 70), text, fill="black")

    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG")
    image_buffer.seek(0)

    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=(500, 180))
    pdf.drawImage(ImageReader(image_buffer), 0, 0, width=500, height=180)
    pdf.showPage()
    pdf.save()
    return pdf_buffer.getvalue()
