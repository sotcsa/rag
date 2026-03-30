"""
Embedding generálás az Ollama bge-m3 modellel.
"""

import logging

import config
from ollama_client import client

logger = logging.getLogger(__name__)


def generate_embeddings(texts: list[str], model: str = None) -> list[list[float]]:
    """
    Szövegek embedding vektorainak generálása.

    A bge-m3 modellt használja, ami 1024 dimenziós vektorokat ad vissza.
    Támogatja a batch feldolgozást.

    Args:
        texts: Szövegek listája
        model: Ollama embedding modell neve

    Returns:
        Embedding vektorok listája
    """
    model = model or config.EMBEDDING_MODEL

    if not texts:
        return []

    logger.info("Embedding generálás: %d szöveg (%s modell)", len(texts), model)

    response = client.embed(model=model, input=texts)
    embeddings = response["embeddings"]

    logger.info("Embedding kész: %d vektor, dimenzió: %d", len(embeddings), len(embeddings[0]))
    return embeddings


def generate_single_embedding(text: str, model: str = None) -> list[float]:
    """
    Egyetlen szöveg embedding vektora.
    Kérdések embedding-jéhez használjuk a keresésnél.
    """
    model = model or config.EMBEDDING_MODEL
    response = client.embed(model=model, input=[text])
    return response["embeddings"][0]
