from __future__ import annotations

from io import BytesIO

from docx import Document

from report_exporters import ReportSection, build_report_docx, build_report_markdown, build_report_pdf


def test_build_report_markdown() -> None:
    markdown = build_report_markdown(
        "Insight Report",
        "paper.pdf",
        [ReportSection("Summary", "Key findings here.")],
    )
    assert "# Insight Report" in markdown
    assert "## Summary" in markdown


def test_build_report_docx() -> None:
    data = build_report_docx(
        "Insight Report",
        "paper.pdf",
        [ReportSection("Summary", "- First point\n- Second point")],
    )
    document = Document(BytesIO(data))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    assert "Insight Report" in text
    assert "First point" in text


def test_build_report_pdf() -> None:
    data = build_report_pdf(
        "Insight Report",
        "paper.pdf",
        [ReportSection("Summary", "Key findings here.")],
    )
    assert data.startswith(b"%PDF")
