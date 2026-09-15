from typing import Any


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = 1000,
    overlap: int = 200,
    min_chunk_size: int = 100,
) -> list[dict[str, Any]]:
    """
    Split extracted PDF pages into overlapping text chunks.

    Each chunk keeps the source page and document name so that
    retrieved answers can later include citations.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if min_chunk_size < 0:
        raise ValueError("min_chunk_size cannot be negative")

    chunks = []

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        start = 0
        chunk_number = 0

        while start < len(text):
            end = start + chunk_size

            # Prefer ending at a natural boundary.
            if end < len(text):
                boundary = text.rfind(" ", start, end)

                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()

            if len(chunk_text) >= min_chunk_size:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page["page"],
                        "source": page["source"],
                        "chunk_id": (
                            f"{page['source']}"
                            f"-p{page['page']}"
                            f"-c{chunk_number}"
                        ),
                    }
                )
                chunk_number += 1

            if end >= len(text):
                break

            start = end - overlap

    return chunks