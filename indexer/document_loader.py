"""
Dokumentum betöltők PDF, DOCX és TXT fájlokhoz.
A cél: strukturált szöveget kinyerni, ami alkalmas LLM-alapú chunkolásra.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """Betöltött dokumentum reprezentációja."""

    content: str
    file_path: str
    file_type: str
    metadata: dict = field(default_factory=dict)


def load_pdf(file_path: Path) -> Document:
    """
    PDF fájl betöltése pymupdf4llm-mel.
    Markdown formátumú kimenetet ad, megőrzi a fejléceket, táblázatokat.
    """
    import pymupdf4llm

    md_text = pymupdf4llm.to_markdown(str(file_path))

    return Document(
        content=md_text,
        file_path=str(file_path),
        file_type="pdf",
        metadata={
            "source": str(file_path),
            "filename": file_path.name,
        },
    )


def load_docx(file_path: Path) -> Document:
    """
    DOCX fájl betöltése python-docx-szel.
    Paragrafusonként, stílusinformációkkal.
    """
    from docx import Document as DocxDocument

    doc = DocxDocument(str(file_path))

    parts = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        # Heading stílusok Markdown fejlécekké alakítása
        style_name = paragraph.style.name if paragraph.style else ""
        if style_name.startswith("Heading 1"):
            parts.append(f"# {text}")
        elif style_name.startswith("Heading 2"):
            parts.append(f"## {text}")
        elif style_name.startswith("Heading 3"):
            parts.append(f"### {text}")
        elif style_name.startswith("Heading"):
            parts.append(f"#### {text}")
        else:
            parts.append(text)

    # Táblázatok feldolgozása
    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append("| " + " | ".join(cells) + " |")

        if rows:
            # Markdown táblázat fejléc elválasztó
            header_sep = "| " + " | ".join(["---"] * len(table.rows[0].cells)) + " |"
            rows.insert(1, header_sep)
            parts.append("\n".join(rows))

    content = "\n\n".join(parts)

    return Document(
        content=content,
        file_path=str(file_path),
        file_type="docx",
        metadata={
            "source": str(file_path),
            "filename": file_path.name,
        },
    )


def load_txt(file_path: Path) -> Document:
    """TXT fájl betöltése UTF-8 kódolással."""
    content = file_path.read_text(encoding="utf-8")

    return Document(
        content=content,
        file_path=str(file_path),
        file_type="txt",
        metadata={
            "source": str(file_path),
            "filename": file_path.name,
        },
    )


def load_document(file_path: Path) -> Document:
    """
    Automatikus fájlformátum felismerés és betöltés.
    Támogatott: .pdf, .docx, .txt
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".txt": load_txt,
        ".md": load_txt,
    }

    loader = loaders.get(suffix)
    if loader is None:
        raise ValueError(
            f"Nem támogatott fájlformátum: {suffix} ({file_path.name}). "
            f"Támogatott: {', '.join(loaders.keys())}"
        )

    return loader(file_path)
