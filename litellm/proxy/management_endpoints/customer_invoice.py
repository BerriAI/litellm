from collections.abc import Sequence
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import Final
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable, Image, LongTable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, TableStyle

from litellm.types.proxy.management_endpoints.common_daily_activity import (
    DailySpendData,
    SpendAnalyticsPaginatedResponse,
)

_BLUE: Final = colors.HexColor("#2b3fd6")
_INK: Final = colors.HexColor("#111827")
_MUTED: Final = colors.HexColor("#64748b")
_BORDER: Final = colors.HexColor("#e2e8f0")
_WIDTH: Final = A4[0] - 88
_BODY: Final = ParagraphStyle("InvoiceBody", fontName="Helvetica", fontSize=9, leading=13, textColor=_INK)
_SMALL: Final = ParagraphStyle("InvoiceSmall", fontName="Helvetica", fontSize=8, leading=12, textColor=_MUTED)
_HEADING: Final = ParagraphStyle(
    "InvoiceHeading", fontName="Helvetica-Bold", fontSize=12, leading=17, textColor=_INK, spaceBefore=18, spaceAfter=10
)
_TITLE: Final = ParagraphStyle("InvoiceTitle", fontName="Helvetica-Bold", fontSize=26, leading=32, textColor=_INK)
_AMOUNT: Final = ParagraphStyle("InvoiceAmount", fontName="Helvetica-Bold", fontSize=27, leading=34, textColor=_BLUE)
_CELL: Final = ParagraphStyle("InvoiceCell", fontName="Helvetica", fontSize=8, leading=12, textColor=_INK)
_NUMBER: Final = ParagraphStyle("InvoiceNumber", parent=_CELL, alignment=TA_RIGHT)
_HEADER: Final = ParagraphStyle("InvoiceHeader", parent=_CELL, textColor=colors.white, fontName="Helvetica-Bold")
_HEADER_NUMBER: Final = ParagraphStyle("InvoiceHeaderNumber", parent=_HEADER, alignment=TA_RIGHT)


def _money(value: Decimal) -> str:
    precision: Final = Decimal("0.000001")
    if 0 < value < precision:
        return "< $0.000001"
    if -precision < value < 0:
        return "> -$0.000001"
    rendered: Final = format(value.quantize(precision, rounding=ROUND_HALF_UP), ",f")
    whole, dot, fraction = rendered.partition(".")
    return f"${whole}.{fraction.rstrip('0').ljust(2, '0')}" if dot else f"${whole}.00"


def _paragraph(text: str, style: ParagraphStyle = _BODY) -> Paragraph:
    return Paragraph(escape(text), style)


def _model_names(days: Sequence[DailySpendData]) -> tuple[str, ...]:
    return tuple(sorted(frozenset(chain.from_iterable(day.breakdown.model_groups for day in days))))


def _model_row(name: str, days: Sequence[DailySpendData]) -> tuple[str, str, str, str, str]:
    entries: Final = tuple(
        day.breakdown.model_groups[name].metrics for day in days if name in day.breakdown.model_groups
    )
    spend: Final = sum((Decimal(str(entry.spend)) for entry in entries), Decimal(0))
    return (
        name,
        f"{sum(entry.api_requests for entry in entries):,}",
        f"{sum(entry.prompt_tokens for entry in entries):,}",
        f"{sum(entry.completion_tokens for entry in entries):,}",
        _money(spend),
    )


def _daily_row(day: date, days: Sequence[DailySpendData]) -> tuple[str, str, str, str]:
    entries: Final = tuple(entry.metrics for entry in days if entry.date == day)
    return (
        day.isoformat(),
        f"{sum(entry.api_requests for entry in entries):,}",
        f"{sum(entry.total_tokens for entry in entries):,}",
        _money(sum((Decimal(str(entry.spend)) for entry in entries), Decimal(0))),
    )


def _table(
    headers: tuple[str, ...], rows: Sequence[tuple[str, ...]], widths: tuple[float, ...], *, compact: bool = False
) -> LongTable:
    cells: Final = tuple(
        tuple(_paragraph(value, _CELL if index == 0 else _NUMBER) for index, value in enumerate(row)) for row in rows
    )
    table: Final = LongTable(
        [
            tuple(_paragraph(value, _HEADER if index == 0 else _HEADER_NUMBER) for index, value in enumerate(headers)),
            *cells,
        ],
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            (
                ("BACKGROUND", (0, 0), (-1, 0), _INK),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), (colors.white, colors.HexColor("#f8fafc"))),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 4 if compact else 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4 if compact else 9),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, _BORDER),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, _BORDER),
            )
        )
    )
    return table


def _page_footer(canvas: Canvas, document: SimpleDocTemplate) -> None:
    canvas.saveState()
    canvas.setStrokeColor(_BORDER)
    canvas.line(44, 45, A4[0] - 44, 45)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(_MUTED)
    canvas.drawString(44, 30, "LiteLLM | Usage invoice | USD")
    canvas.drawRightString(A4[0] - 44, 30, f"Page {document.page}")
    canvas.restoreState()


def _summary(
    customer_id: str, alias: str | None, period: tuple[date, date], issued_at: datetime, days: Sequence[DailySpendData]
) -> tuple[Flowable, ...]:
    start, end = period
    total: Final = sum((Decimal(str(day.metrics.spend)) for day in days), Decimal(0))
    logo: Final = Path(__file__).resolve().parents[1] / "logo.png"
    return (
        Image(str(logo), width=108, height=30, kind="proportional", hAlign="LEFT"),
        Spacer(1, 24),
        _paragraph("Usage invoice", _TITLE),
        Spacer(1, 8),
        _paragraph(f"{start.isoformat()} - {end.isoformat()} (inclusive)", _SMALL),
        Spacer(1, 22),
        _paragraph(alias or customer_id, _HEADING),
        _paragraph(f"Customer ID  {customer_id}", _SMALL),
        _paragraph(f"Issued  {issued_at.strftime('%Y-%m-%d %H:%M UTC')}", _SMALL),
        Spacer(1, 24),
        _paragraph("Total usage", _SMALL),
        _paragraph(_money(total), _AMOUNT),
        Spacer(1, 12),
        _paragraph(
            f"{sum(day.metrics.api_requests for day in days):,} requests  |  "
            f"{sum(day.metrics.total_tokens for day in days):,} tokens  |  Currency USD",
            _BODY,
        ),
        _paragraph("Model breakdown", _HEADING),
        _paragraph("Recorded usage across all keys for this customer, grouped by serving model group.", _SMALL),
        Spacer(1, 10),
    )


def render_customer_invoice(
    customer_id: str,
    alias: str | None,
    period: tuple[date, date],
    activity: SpendAnalyticsPaginatedResponse,
    issued_at: datetime,
) -> bytes:
    days: Final = tuple(activity.results)
    names: Final = _model_names(days)
    model_rows: Final = tuple(_model_row(name, days) for name in names)
    start, end = period
    dates: Final = tuple(start + timedelta(days=index) for index in range((end - start).days + 1))
    daily_rows: Final = tuple(_daily_row(day, days) for day in dates)
    model_table: Final = (
        _table(
            ("Model group", "Requests", "Input tokens", "Output tokens", "Cost (USD)"),
            model_rows,
            (_WIDTH * 0.34, _WIDTH * 0.14, _WIDTH * 0.17, _WIDTH * 0.17, _WIDTH * 0.18),
        )
        if model_rows
        else _paragraph("No usage was recorded for this customer in this period.", _BODY)
    )
    story: Final = (
        *_summary(customer_id, alias, period, issued_at, days),
        model_table,
        PageBreak(),
        _paragraph("Daily usage", _HEADING),
        _paragraph(f"Customer ID  {customer_id}  |  {start.isoformat()} - {end.isoformat()}", _SMALL),
        Spacer(1, 12),
        _table(
            ("Date", "Requests", "Tokens", "Cost (USD)"),
            daily_rows,
            (_WIDTH * 0.34, _WIDTH * 0.2, _WIDTH * 0.23, _WIDTH * 0.23),
            compact=True,
        ),
    )
    output: Final = BytesIO()
    document: Final = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=44,
        leftMargin=44,
        topMargin=44,
        bottomMargin=62,
        title="LiteLLM Usage invoice",
        author="LiteLLM",
    )
    document.build(list(story), onFirstPage=_page_footer, onLaterPages=_page_footer)
    return output.getvalue()
