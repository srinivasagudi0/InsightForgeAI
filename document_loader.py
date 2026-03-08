from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path


class DocumentLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedDocument:
    name: str
    content: str
    source_type: str
    word_count: int
    char_count: int
    page_count: int | None = None
    used_ocr: bool = False


def load_document(file_path: str) -> LoadedDocument:
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise DocumentLoadError("The provided file path does not exist.")

    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(f"Unable to read file: {exc}") from exc

    return load_uploaded_document(path.name, data)


def load_uploaded_document(file_name: str, data: bytes) -> LoadedDocument:
    suffix = Path(file_name).suffix.lower()

    if suffix == ".pdf":
        content, page_count, used_ocr = _load_pdf_bytes(data)
        return _build_loaded_document(
            name=file_name,
            content=content,
            source_type="PDF",
            page_count=page_count,
            used_ocr=used_ocr,
        )

    content = _load_text_bytes(data)
    return _build_loaded_document(
        name=file_name,
        content=content,
        source_type="Text",
    )


def load_text_input(name: str, text: str) -> LoadedDocument:
    return _build_loaded_document(
        name=name,
        content=text,
        source_type="Text",
    )


def _build_loaded_document(
    name: str,
    content: str,
    source_type: str,
    page_count: int | None = None,
    used_ocr: bool = False,
) -> LoadedDocument:
    normalized = _normalize_content(content)
    if not normalized:
        raise DocumentLoadError(
            "The document is empty or contains no extractable text."
        )

    return LoadedDocument(
        name=name,
        content=normalized,
        source_type=source_type,
        page_count=page_count,
        used_ocr=used_ocr,
        word_count=len(normalized.split()),
        char_count=len(normalized),
    )


def _load_text_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _load_pdf_bytes(data: bytes) -> tuple[str, int, bool]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentLoadError(
            "PDF support requires pypdf. Install it with 'pip install pypdf'."
        ) from exc

    try:
        reader = PdfReader(BytesIO(data))
    except Exception as exc:
        raise DocumentLoadError(f"Unable to open PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            decrypt_result = reader.decrypt("")
        except Exception as exc:
            raise DocumentLoadError("Encrypted PDF files are not supported.") from exc
        if decrypt_result == 0:
            raise DocumentLoadError("Encrypted PDF files are not supported.")

    page_count = len(reader.pages)
    page_text: dict[int, str] = {}
    missing_pages: list[int] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:
            raise DocumentLoadError(
                f"Unable to extract text from PDF page {page_number}: {exc}"
            ) from exc

        if text:
            page_text[page_number] = text
        else:
            missing_pages.append(page_number)

    used_ocr = False
    if missing_pages:
        ocr_results = _ocr_pdf_pages(
            data,
            missing_pages,
            fail_if_unavailable=not page_text,
        )
        if ocr_results:
            used_ocr = True
            page_text.update(ocr_results)

    pages = [
        f"[Page {page_number}]\n{page_text[page_number]}"
        for page_number in range(1, page_count + 1)
        if page_number in page_text and page_text[page_number].strip()
    ]

    if not pages:
        raise DocumentLoadError(
            "No extractable text was found in the PDF. Install OCR support or use a text-based PDF."
        )

    return "\n\n".join(pages), page_count, used_ocr


def _ocr_pdf_pages(
    data: bytes,
    page_numbers: list[int],
    fail_if_unavailable: bool,
) -> dict[int, str]:
    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        if fail_if_unavailable:
            raise DocumentLoadError(
                "No extractable text was found in the PDF and tesseract is not installed for OCR."
            )
        return {}

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        if fail_if_unavailable:
            raise DocumentLoadError(
                "No extractable text was found in the PDF. Install 'pypdfium2' to enable OCR for scanned PDFs."
            ) from exc
        return {}

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        if fail_if_unavailable:
            raise DocumentLoadError(
                "Pillow is required for OCR image preprocessing."
            ) from exc
        return {}

    ocr_results: dict[int, str] = {}
    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as exc:
        if fail_if_unavailable:
            raise DocumentLoadError(
                f"Unable to initialize OCR rendering for the PDF: {exc}"
            ) from exc
        return {}

    for page_number in page_numbers:
        try:
            page = pdf[page_number - 1]
            pil_image = page.render(scale=3).to_pil()
        except Exception:
            continue

        processed_image = _prepare_image_for_ocr(pil_image, Image, ImageOps)
        text = _run_tesseract_ocr(processed_image, tesseract_path)
        if text:
            ocr_results[page_number] = text

    if fail_if_unavailable and not ocr_results:
        raise DocumentLoadError(
            "OCR could not extract text from this scanned PDF."
        )

    return ocr_results


def _prepare_image_for_ocr(image, image_module, image_ops_module) -> object:
    grayscale = image.convert("L")
    normalized = image_ops_module.autocontrast(grayscale)
    if normalized.width < 1200:
        normalized = normalized.resize(
            (max(1, normalized.width * 2), max(1, normalized.height * 2)),
            image_module.Resampling.LANCZOS,
        )
    return normalized


def _run_tesseract_ocr(image, tesseract_path: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png") as temp_image:
        image.save(temp_image.name)
        result = subprocess.run(
            [tesseract_path, temp_image.name, "stdout", "--psm", "6"],
            capture_output=True,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _normalize_content(content: str) -> str:
    lines = [line.rstrip() for line in content.replace("\x00", "").splitlines()]
    normalized = "\n".join(lines).strip()
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized
