from pathlib import Path

from pypdf import PdfReader


def load_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF while preserving page numbers.

    Returns:
        A list of dictionaries, one per page:
        [
            {
                "text": "...",
                "page": 1,
                "source": "document.pdf"
            }
        ]
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = PdfReader(str(path))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        text = text.strip()

        if text:
            pages.append(
                {
                    "text": text,
                    "page": page_number,
                    "source": path.name,
                }
            )

    return pages