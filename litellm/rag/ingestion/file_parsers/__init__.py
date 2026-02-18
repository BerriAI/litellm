"""
File parsers for RAG ingestion.

Provides text extraction utilities for various file formats.
"""

from .pdf_parser import extract_text_from_pdf

__all__ = ["extract_text_from_pdf"]
