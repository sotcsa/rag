"""
LLM-alapú szemantikus chunkolás.
A qwen2.5:14b modellt használja a dokumentumok intelligens szegmentálásához.
Fallback: rekurzív karakter-alapú chunkolás, ha az LLM nem elérhető.
"""

import json
import logging
import re
from dataclasses import dataclass, field

import config
from ollama_client import client

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
7. KRITIKUS: MINDEN belső idézőjelet (") escape-elj (\\") a 'content' és 'summary' értékeken belül!

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

    def _validate(data):
        """Ellenőrzi, hogy a parse-olt adat megfelelő formátumú-e."""
        if isinstance(data, list) and len(data) > 0 and all(
            isinstance(item, dict) and "content" in item and "summary" in item
            for item in data
        ):
            return data
        return None

    def _try_parse(text):
        """Próbálja parse-olni a szöveget JSON-ként."""
        try:
            return _validate(json.loads(text))
        except (json.JSONDecodeError, TypeError):
            return None

    def _fix_json_strings(text):
        """
        Az LLM gyakran érvénytelen escape szekvenciákat tesz a JSON string-ekbe:
        - Literal sortörések (newline helyett valódi newline)
        - Markdown escape-ek: backslash+( backslash+) stb. (nem valid JSON escape)
        Ezeket javítjuk, hogy valid JSON legyen.
        """
        fixed = []
        in_string = False
        i = 0
        while i < len(text):
            ch = text[i]

            if not in_string:
                if ch == '"':
                    in_string = True
                fixed.append(ch)
                i += 1
                continue

            # String-en belül vagyunk
            if ch == '"':
                # String vége, vagy belső idézőjel?
                # A JSON string végét a kulcsoknál ':' a valuesoknál ',', '}', vagy ']' követheti.
                lookahead = ""
                for j in range(i + 1, len(text)):
                    if not text[j].isspace():
                        lookahead = text[j]
                        break
                
                if lookahead in (':', ',', '}', ']', ''):
                    # Valódi string vége
                    in_string = False
                    fixed.append(ch)
                    i += 1
                    continue
                else:
                    # Belső, escapálatlan idézőjel! Javítjuk.
                    fixed.append('\\"')
                    i += 1
                    continue

            if ch == '\\' and i + 1 < len(text):
                next_ch = text[i + 1]
                # Valid JSON escape szekvenciák: \" \\ \/ \b \f \n \r \t \uXXXX
                if next_ch in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'):
                    fixed.append(ch)
                    fixed.append(next_ch)
                    i += 2
                    continue
                else:
                    # Érvénytelen escape → dupla backslash-re cseréljük: \( → \\(
                    fixed.append('\\\\')
                    i += 1
                    continue

            if ch == '\n':
                fixed.append('\\n')
                i += 1
                continue

            if ch == '\r':
                i += 1
                continue

            if ch == '\t':
                fixed.append('\\t')
                i += 1
                continue

            fixed.append(ch)
            i += 1

        return ''.join(fixed)

    # 0. JSON string-ek javítása (érvénytelen escape szekvenciák + literal newline-ok)
    fixed_text = _fix_json_strings(response_text)

    # 1. Közvetlen parse (javított szöveggel)
    result = _try_parse(fixed_text)
    if result:
        return result

    # 2. Markdown code block-ban keresés (```json ... ``` vagy ``` ... ```)
    code_block_matches = re.findall(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", fixed_text)
    for match in code_block_matches:
        result = _try_parse(match)
        if result:
            return result

    # 3. JSON tömb keresés a szövegben (legkülső [ ... ] pár)
    bracket_depth = 0
    start_idx = None
    for i, ch in enumerate(fixed_text):
        if ch == '[':
            if bracket_depth == 0:
                start_idx = i
            bracket_depth += 1
        elif ch == ']':
            bracket_depth -= 1
            if bracket_depth == 0 and start_idx is not None:
                candidate = fixed_text[start_idx:i + 1]
                result = _try_parse(candidate)
                if result:
                    return result

    # 4. Trailing comma javítás és újrapróbálás
    cleaned = re.sub(r",\s*([}\]])", r"\1", fixed_text)
    result = _try_parse(cleaned)
    if result:
        return result

    # 5. Ha semmi nem működött, logoljuk a választ
    logger.warning(
        "JSON parse sikertelen. LLM válasz (első 300 karakter): %.300s",
        response_text,
    )
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
    progress_callback=None,
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
            
            if progress_callback:
                progress_callback(seg_idx, len(segments))
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "--- LLM KÉRÉS (Szegmens %d/%d) ---\n%s\n"
                    "----------------------------------",
                    seg_idx + 1, len(segments), prompt
                )

            if model.startswith("openrouter/"):
                import urllib.request
                import json
                or_model = model.replace("openrouter/", "")
                api_key = getattr(config, "OPENROUTER_API_KEY", "")
                if not api_key:
                    raise ValueError("OPENROUTER_API_KEY nincs beállítva a config.py-ban!")
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "http://localhost:8000",
                    "Content-Type": "application/json"
                }
                data = json.dumps({
                    "model": or_model,
                    "messages": [
                        {"role": "system", "content": CHUNKING_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": config.LLM_TEMPERATURE,
                }).encode("utf-8")
                
                req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=data, headers=headers)
                with urllib.request.urlopen(req) as resp:
                    res_body = resp.read().decode("utf-8")
                    res_json = json.loads(res_body)
                    response_text = res_json["choices"][0]["message"]["content"]
            else:
                response = client.chat(
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
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "--- LLM VÁLASZ (Szegmens %d/%d) ---\n%s\n"
                    "-----------------------------------",
                    seg_idx + 1, len(segments), response_text
                )

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
