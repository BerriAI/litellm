from __future__ import annotations

import base64
from functools import cache
from io import BytesIO
from typing import Final
from urllib.parse import quote

from PIL import Image, ImageDraw
from reportlab.graphics.barcode import code128  # pyright: ignore[reportMissingTypeStubs]  # ReportLab has no stubs
from reportlab.lib import colors  # pyright: ignore[reportMissingTypeStubs]  # ReportLab has no stubs
from reportlab.lib.pagesizes import letter  # pyright: ignore[reportMissingTypeStubs]  # ReportLab has no stubs
from reportlab.lib.utils import ImageReader  # pyright: ignore[reportMissingTypeStubs]  # ReportLab has no stubs
from reportlab.pdfgen import canvas  # pyright: ignore[reportMissingTypeStubs]  # ReportLab has no stubs


def dummy_image_url(text: str, font_size: int, width: int = 800, height: int = 300) -> str:
    return f"https://dummyjson.com/image/{width}x{height}/ffffff/000000?text={quote(text)}&fontSize={font_size}"


_GLYPHS: Final = {
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
}


@cache
def structured_image_bytes() -> bytes:
    image: Final = Image.new("RGB", (320, 80), "white")
    draw: Final = ImageDraw.Draw(image)
    scale: Final = 8
    cursor_x = 24
    for character in "OCR 123":
        if character == " ":
            cursor_x += scale * 3
            continue
        for glyph_y, row in enumerate(_GLYPHS[character]):
            for glyph_x, filled in enumerate(row):
                if filled == "1":
                    x = cursor_x + glyph_x * scale
                    y = 12 + glyph_y * scale
                    draw.rectangle((x, y, x + scale - 1, y + scale - 1), fill="black")
        cursor_x += scale * 6
    output: Final = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@cache
def structured_image_data_uri() -> str:
    encoded: Final = base64.b64encode(structured_image_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _draw_header(pdf: canvas.Canvas, title: str, page_number: int) -> None:
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 11)
    pdf.drawString(45, 770, "Quarterly Operations Report")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(45, 745, title)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(45, 30, f"Confidential | Page {page_number} of 5")


def _draw_body(pdf: canvas.Canvas, page_number: int) -> None:
    pdf.setFont("Helvetica", 10)
    for line_number in range(1, 9):
        pdf.drawString(
            45,
            500 - (line_number * 28),
            f"Section {page_number}.{line_number}: Invoice totals, regional revenue, and reconciliation notes.",
        )


def _diagram_image(width: int, height: int, accent: tuple[int, int, int]) -> Image.Image:
    image: Final = Image.new("RGB", (width, height), (242, 246, 252))
    draw: Final = ImageDraw.Draw(image)
    for coordinate in range(0, max(width, height), 40):
        draw.line((coordinate, 0, coordinate, height), fill=(32, 32, 32), width=3)
        draw.line((0, coordinate, width, coordinate), fill=(32, 32, 32), width=3)
    draw.line((0, 0, width, height), fill=accent, width=8)
    draw.line((width, 0, 0, height), fill=accent, width=8)
    draw.rectangle((width // 4, height // 4, width * 3 // 4, height * 3 // 4), outline=accent, width=6)
    return image


def _draw_embedded_images(pdf: canvas.Canvas) -> None:
    images: Final = (
        (_diagram_image(320, 320, (51, 115, 217)), 455, 655, 70, 70),
        (_diagram_image(360, 320, (38, 151, 92)), 455, 565, 70, 62),
        (_diagram_image(120, 120, (219, 68, 55)), 455, 500, 45, 45),
    )
    for image, x, y, width, height in images:
        pdf.drawImage(  # pyright: ignore[reportUnknownMemberType]  # ReportLab has no stubs
            ImageReader(image), x, y, width=width, height=height, mask="auto"
        )


def _draw_table_page(pdf: canvas.Canvas) -> None:
    columns: Final = (45, 245, 405, 565)
    tables: Final = (
        (
            (730, 695, 660, 625),
            (
                ("Item", "Quantity", "Amount", 707),
                ("Document analysis", "2", "120.00", 672),
                ("OCR verification", "1", "80.00", 637),
            ),
        ),
        (
            (600, 565, 530, 495),
            (
                ("Item continued", "Quantity", "Amount", 577),
                ("Fixture validation", "3", "45.00", 542),
                ("Provider review", "1", "25.00", 507),
            ),
        ),
    )
    for rows, values in tables:
        for x in columns:
            pdf.line(x, rows[-1], x, rows[0])
        for y in rows:
            pdf.line(45, y, 565, y)
        for item, quantity, amount, y in values:
            pdf.drawString(55, y, item)
            pdf.drawString(255, y, quantity)
            pdf.drawString(415, y, amount)


def _draw_chart_page(pdf: canvas.Canvas) -> None:
    bars: Final = ((70, 70), (170, 115), (270, 90), (370, 130))
    pdf.setFillColor(colors.HexColor("#3373D9"))
    for x, height in bars:
        pdf.rect(x, 610, 65, height, fill=1, stroke=0)
    pdf.setFillColor(colors.black)
    for quarter, x in zip(("Q1", "Q2", "Q3", "Q4"), (90, 190, 290, 390), strict=True):
        pdf.drawString(x, 590, quarter)
    pdf.drawString(45, 550, "Formula: gross margin = (revenue - cost) / revenue")
    _draw_embedded_images(pdf)


def _draw_metadata_page(pdf: canvas.Canvas) -> None:
    pdf.setFont("Helvetica", 12)
    pdf.drawString(45, 700, "Invoice Number: INV-2048")
    pdf.drawString(45, 675, "Purchase Order: PO-4096")
    pdf.setFillColor(colors.HexColor("#F2E65A"))
    pdf.rect(40, 555, 500, 24, fill=1, stroke=0)
    pdf.setFillColor(colors.black)
    pdf.drawString(45, 560, "Highlighted total requiring review")
    pdf.drawString(45, 530, "Reviewer comment: verify the highlighted total before approval")
    pdf.setFillColor(colors.red)
    pdf.drawString(45, 495, "Revised total: 245.00")
    pdf.line(45, 501, 150, 501)
    pdf.setFillColor(colors.black)
    pdf.linkURL(  # pyright: ignore[reportUnknownMemberType]  # ReportLab has no stubs
        "https://example.com/invoices/INV-2048", (45, 575, 300, 590), relative=0
    )
    pdf.highlightAnnotation(  # pyright: ignore[reportUnknownMemberType]  # ReportLab has no stubs
        "Total highlighted for review",
        Rect=(40, 555, 540, 579),
        QuadPoints=(40, 579, 540, 579, 40, 555, 540, 555),
    )
    pdf.textAnnotation(  # pyright: ignore[reportUnknownMemberType]  # ReportLab has no stubs
        "Verify the highlighted total", Rect=(520, 525, 540, 545)
    )
    pdf.drawString(45, 575, "https://example.com/invoices/INV-2048")
    barcode: Final = code128.Code128("5901234123457", barHeight=70, barWidth=1.2)
    barcode.drawOn(pdf, 90, 130)


def _draw_signature_page(pdf: canvas.Canvas) -> None:
    pdf.saveState()
    pdf.setFillColor(colors.lightgrey)
    pdf.setFont("Helvetica-Bold", 54)
    pdf.translate(110, 390)
    pdf.rotate(25)
    pdf.drawString(0, 0, "DRAFT")
    pdf.restoreState()
    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 12)
    pdf.drawString(45, 635, "Approved by: Jordan Lee")
    pdf.line(45, 610, 310, 610)
    pdf.bezier(55, 595, 75, 625, 112, 602, 155, 600)
    pdf.drawString(45, 580, "Signature")


def _draw_appendix_page(pdf: canvas.Canvas) -> None:
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(45, 700, "1. Scope")
    pdf.drawString(45, 650, "2. Findings")
    pdf.drawString(45, 600, "3. Recommendations")


@cache
def structured_pdf_bytes() -> bytes:
    output: Final = BytesIO()
    pdf: Final = canvas.Canvas(output, pagesize=letter, pageCompression=0, invariant=1)
    pdf.setTitle("Quarterly Operations Report")
    pdf.setAuthor("LiteLLM OCR fixture generator")
    pdf.setSubject("Semantic OCR coverage for tables, figures, annotations, and metadata")
    pdf.setKeywords("OCR, invoice, table, figure, annotation")
    pages: Final = (
        ("Invoice Summary and Line Items", _draw_table_page),
        ("Revenue Chart and Formula Review", _draw_chart_page),
        ("Key Values, Link, Highlight, and Comment", _draw_metadata_page),
        ("Approval Signature and Watermark", _draw_signature_page),
        ("Appendix with Section Boundaries", _draw_appendix_page),
    )
    for page_number, (title, draw_page) in enumerate(pages, start=1):
        _draw_header(pdf, title, page_number)
        draw_page(pdf)
        _draw_body(pdf, page_number)
        pdf.showPage()
    pdf.save()
    return output.getvalue()


@cache
def structured_pdf_data_uri() -> str:
    encoded: Final = base64.b64encode(structured_pdf_bytes()).decode("ascii")
    return f"data:application/pdf;base64,{encoded}"
