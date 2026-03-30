"""
RAG válaszgenerálás LLM-mel.
A megtalált kontextus chunkok és a felhasználói kérdés alapján generál választ.
"""

import logging

import ollama

import config

logger = logging.getLogger(__name__)


RAG_SYSTEM_PROMPT = """Te egy segítőkész magyar nyelvű asszisztens vagy. A feladatod, hogy a megadott kontextus alapján válaszolj a felhasználó kérdésére.

Szabályok:
1. KIZÁRÓLAG a megadott kontextus alapján válaszolj
2. Ha a kontextus nem tartalmaz elegendő információt a válaszhoz, mondd el őszintén
3. Magyarul válaszolj, még akkor is, ha a kontextus angol nyelvű
4. Hivatkozz a forrásokra a válaszodban (pl. "A [forrás neve] szerint...")
5. Légy tömör és informatív"""

RAG_USER_PROMPT = """Kontextus a dokumentumokból:

{context}

---

Kérdés: {query}

Válaszolj a fenti kontextus alapján, magyarul:"""

NO_CONTEXT_RESPONSE = (
    "Sajnos nem találtam releváns információt a dokumentumok között ehhez a kérdéshez. "
    "Próbáld átfogalmazni a kérdést, vagy ellenőrizd, hogy a megfelelő dokumentumok indexelve vannak-e."
)


def generate_answer(
    query: str,
    context: str,
    model: str = None,
    stream: bool = False,
):
    """
    RAG válasz generálása.

    Args:
        query: Felhasználói kérdés
        context: A retriever által összeállított kontextus szöveg
        model: Ollama modell neve
        stream: Ha True, generator-ként adja vissza a tokeneket (streaming)

    Returns:
        Ha stream=False: teljes válasz szöveg (str)
        Ha stream=True: token generator
    """
    model = model or config.LLM_MODEL

    if not context:
        if stream:
            def _empty_gen():
                yield NO_CONTEXT_RESPONSE
            return _empty_gen()
        return NO_CONTEXT_RESPONSE

    prompt = RAG_USER_PROMPT.format(context=context, query=query)

    logger.info("Válasz generálás (modell: %s, stream: %s)", model, stream)

    if stream:
        return _stream_response(model, prompt)
    else:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.3,
                "num_ctx": config.LLM_NUM_CTX,
            },
        )
        return response["message"]["content"]


def _stream_response(model: str, prompt: str):
    """Token-enként streamed válasz generálás."""
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0.3,
            "num_ctx": config.LLM_NUM_CTX,
        },
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]
