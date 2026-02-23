"""
PDF text extraction utilities.

Provides text extraction from PDF files using pypdf or PyPDF2.
"""

from typing import Optional

from litellm._logging import verbose_logger


def extract_text_from_pdf(file_content: bytes) -> Optional[str]:
    """
    Extract text from PDF using pypdf if available.

    Args:
        file_content: Raw PDF bytes

    Returns:
        Extracted text or None if extraction fails
    """
    try:
        from io import BytesIO

        # Try pypdf first (most common)
        try:
            from pypdf import PdfReader as PypdfReader
            
            pdf_file = BytesIO(file_content)
            reader = PypdfReader(pdf_file)
            
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            if text_parts:
                extracted_text = "\n\n".join(text_parts)
                verbose_logger.debug(f"Extracted {len(extracted_text)} characters from PDF using pypdf")
                return extracted_text
                
        except ImportError:
            verbose_logger.debug("pypdf not available, trying PyPDF2")
            
        # Fallback to PyPDF2
        try:
            from PyPDF2 import PdfReader as PyPDF2Reader
            
            pdf_file = BytesIO(file_content)
            reader = PyPDF2Reader(pdf_file)
            
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            if text_parts:
                extracted_text = "\n\n".join(text_parts)
                verbose_logger.debug(f"Extracted {len(extracted_text)} characters from PDF using PyPDF2")
                return extracted_text
                
        except ImportError:
            verbose_logger.debug("PyPDF2 not available, PDF extraction requires OCR or pypdf/PyPDF2 library")
            
    except Exception as e:
        verbose_logger.debug(f"PDF text extraction failed: {e}")
        
    return None
