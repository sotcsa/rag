"""
LLM-alapú szemantikus chunkolás.
A qwen2.5:14b modellt használja a dokumentumok intelligens szegmentálásához.
Fallback: rekurzív karakter-alapú chunkolás, ha az LLM nem elérhető.
"""

import json
import logging
import re
from dataclasses import dataclass, field

import ollama

import config

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Egy szemantikus chunk reprezentációja."""

    content: str
    summary: str
    chunk_index: int
    source_file: str
    metadata: dict = field(default_factory=dict)


# --- Magyar nyelvű chunkolási prompt ---
CHUNKING_SYSTEM_PROMPT = """Te egy dokumentum-feldolgozó asszisztens vagy. A feladatod szövegek logikai egységekre bontása.
Mindig pontosan a megadott JSON formátumban válaszolj. Semmilyen egyéb szöveget ne adj hozzá a válaszhoz."""

CHUNKING_USER_PROMPT = """Feladat: Az alábbi szöveget bontsd logikailag összefüggő, önálló egységekre (chunk-okra).

Szabályok:
1. Minden egység legyen 150-600 szó között
2. Minden egységnek legyen egyértelmű, jól körülhatárolható témája
3. Ne vágj el mondatokat, bekezdéseket vagy gondolati egységeket
4. Kapcsolódó információkat tarts együtt (pl. definíció + példa, kérdés + válasz)
5. Minden egységhez adj egy tömör, 1 mondatos magyar összefoglalót
6. Ha a szöveg rövid (kevesebb mint 150 szó), adj vissza egyetlen chunkot

Válaszolj KIZÁRÓLAG érvényes JSON formátumban, semmilyen más szöveget ne írj:
[{{"summary": "Egymondatos összefoglaló", "content": "A chunk teljes szövege..."}}, ...]

Szöveg:
{text}"""


def _pre_segment(text: str, max_size: int = None, overlap: int = None) -> list[str]:
    """
    Szöveg előszegmentálása nagyobb blokkokra, bekezdéshatárok mentén.
    Ez biztosítja, hogy az LLM-nek kezelhető méretű inputot adjunk.
    """
    max_size = max_size or config.PRE_SEGMENT_SIZE
    overlap = overlap or config.PRE_SEGMENT_OVERLAP

    if len(text) <= max_size:
        return [text]

    # Bekezdések mentén próbálunk vágni
    paragraphs = re.split(r"\n\s*\n", text)
    segments = []
    current_segment = ""

    for para in paragraphs:
        if len(current_segment) + len(para) + 2 > max_size and current_segment:
            segments.append(current_segment.strip())
            # Átfedés: az utolsó 'overlap' karakter átkerül a következő szegmensbe
            if overlap > 0 and len(current_segment) > overlap:
                current_segment = current_segment[-overlap:] + "\n\n" + para
            else:
                current_segment = para
        else:
            current_segment = (
                current_segment + "\n\n" + para if current_segment else para
            )

    if current_segment.strip():
        segments.append(current_segment.strip())

    return segments


def _parse_llm_chunks(response_text: str) -> list[dict] | None:
    """
    LLM válasz JSON-jának parse-olása.
    Robusztus: megpróbálja kinyerni a JSON-t akkor is, ha extra szöveg veszi körül.
    """
    # Először próbáljuk közvetlenül
    try:
        result = json.loads(response_text)
        if isinstance(result, list) and all(
            "content" in item and "summary" in item for item in result
        ):
            return result
    except json.JSONDecodeError:
        pass

    # Ha nem sikerült, keressük meg a JSON tömböt a szövegben
    # Keressünk [ ... ] mintát
    json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if isinstance(result, list) and all(
                "content" in item and "summary" in item for item in result
            ):
                return result
        except json.JSONDecodeError:
            pass

    # Markdown code block-ban lévő JSON
    code_block_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", response_text, re.DOTALL)
    if code_block_match:
        try:
            result = json.loads(code_block_match.group(1))
            if isinstance(result, list) and all(
                "content" in item and "summary" in item for item in result
            ):
                return result
        except json.JSONDecodeError:
            pass

    return None


def _fallback_chunk(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
) -> list[dict]:
    """
    Fallback: rekurzív karakter-alapú chunkolás.
    Bekezdés- és mondathatárok mentén vág.
    """
    chunk_size = chunk_size or config.FALLBACK_CHUNK_SIZE
    overlap = overlap or config.FALLBACK_CHUNK_OVERLAP

    if len(text) <= chunk_size:
        return [{"summary": "", "content": text.strip()}]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Próbáljunk bekezdéshatáron vágni
            newline_pos = text.rfind("\n\n", start + chunk_size // 2, end)
            if newline_pos != -1:
                end = newline_pos
            else:
                # Próbáljunk mondathatáron vágni
                sentence_end = max(
                    text.rfind(". ", start + chunk_size // 2, end),
                    text.rfind("! ", start + chunk_size // 2, end),
                    text.rfind("? ", start + chunk_size // 2, end),
                )
                if sentence_end != -1:
                    end = sentence_end + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append({"summary": "", "content": chunk_text})

        start = end - overlap if end < len(text) else len(text)

    return chunks


def chunk_document_with_llm(
    text: str,
    source_file: str,
    model: str = None,
) -> list[Chunk]:
    """
    Dokumentum szöveg LLM-alapú chunkolása.

    Folyamat:
    1. Előszegmentálás ~3000 karakteres blokkokra (bekezdéshatárok mentén)
    2. Minden blokk LLM-es elemzése → logikai egységek azonosítása
    3. JSON parse + validáció
    4. Fallback, ha az LLM nem ad használható választ

    Args:
        text: A dokumentum szövege
        source_file: Forrásfájl útvonala (metaadat)
        model: Ollama modell neve (alapértelmezett: config.LLM_MODEL)

    Returns:
        Chunk objektumok listája
    """
    model = model or config.LLM_MODEL
    all_chunks: list[Chunk] = []
    chunk_index = 0

    # 1. Előszegmentálás
    segments = _pre_segment(text)
    logger.info(
        "Dokumentum előszegmentálva: %d szegmens (%s)",
        len(segments),
        source_file,
    )

    for seg_idx, segment in enumerate(segments):
        # Túl rövid szegmensek → közvetlenül chunk-ként kezelés
        word_count = len(segment.split())
        if word_count < 50:
            all_chunks.append(
                Chunk(
                    content=segment.strip(),
                    summary="",
                    chunk_index=chunk_index,
                    source_file=source_file,
                )
            )
            chunk_index += 1
            continue

        # 2. LLM-es chunkolás
        try:
            prompt = CHUNKING_USER_PROMPT.format(text=segment)
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": CHUNKING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": config.LLM_TEMPERATURE,
                    "num_ctx": config.LLM_NUM_CTX,
                },
            )

            response_text = response["message"]["content"]
            parsed = _parse_llm_chunks(response_text)

            if parsed:
                logger.info(
                    "  Szegmens %d/%d: LLM %d chunk-ot azonosított",
                    seg_idx + 1,
                    len(segments),
                    len(parsed),
                )
                for item in parsed:
                    content = item.get("content", "").strip()
                    if content:
                        all_chunks.append(
                            Chunk(
                                content=content,
                                summary=item.get("summary", ""),
                                chunk_index=chunk_index,
                                source_file=source_file,
                            )
                        )
                        chunk_index += 1
            else:
                # JSON parse sikertelen → fallback
                logger.warning(
                    "  Szegmens %d/%d: LLM válasz nem parse-olható, fallback chunkolás",
                    seg_idx + 1,
                    len(segments),
                )
                fallback_items = _fallback_chunk(segment)
                for item in fallback_items:
                    all_chunks.append(
                        Chunk(
                            content=item["content"],
                            summary=item.get("summary", ""),
                            chunk_index=chunk_index,
                            source_file=source_file,
                        )
                    )
                    chunk_index += 1

        except Exception as e:
            # LLM hiba → fallback
            logger.warning(
                "  Szegmens %d/%d: LLM hiba (%s), fallback chunkolás",
                seg_idx + 1,
                len(segments),
                str(e),
            )
            fallback_items = _fallback_chunk(segment)
            for item in fallback_items:
                all_chunks.append(
                    Chunk(
                        content=item["content"],
                        summary=item.get("summary", ""),
                        chunk_index=chunk_index,
                        source_file=source_file,
                    )
                )
                chunk_index += 1

    logger.info(
        "Chunkolás kész: %d chunk (%s)",
        len(all_chunks),
        source_file,
    )
    return all_chunks
