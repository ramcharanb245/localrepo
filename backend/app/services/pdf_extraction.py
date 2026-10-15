
from io import BytesIO

from pypdf import PdfReader


class EmptyPdfError(Exception):
    """Raised when a PDF contains no extractable text."""


class PdfExtractionError(Exception):
    """Raised when a PDF cannot be read."""


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF held in memory."""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))

        if reader.is_encrypted:
            raise PdfExtractionError("Encrypted PDFs are not supported.")

        extracted_pages = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_pages.append(text)

        extracted_text = "\n".join(extracted_pages).strip()

        if not extracted_text:
            raise EmptyPdfError("PDF contains no extractable text.")

        return extracted_text

    except (EmptyPdfError, PdfExtractionError):
        raise
    except Exception as exc:
        raise PdfExtractionError("Failed to extract text from PDF.") from exc
