"""
Descarga e ingesta TODAS las páginas de Wikipedia bajo la categoría
'Colombia' y sus sub-categorías inmediatas (ciudades, cultura, política…).

Ejemplo de uso:
    python scripts/ingest_colombia_wiki.py   # tarda ±15-20 min la 1.ª vez
"""

import sys, datetime, re, time
from typing import List, Set
import wikipedia

from chromadb import Client
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---- Config ----
COLLECTION   = "colombia_documents"
CHUNK_SIZE   = 1000
CHUNK_OVERLAP = 200
EMB_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
WIKI_LANG    = "es"

wikipedia.set_lang(WIKI_LANG)
client = Client()
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(EMB_MODEL)
col    = client.get_or_create_collection(COLLECTION, embedding_function=emb_fn)
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                          chunk_overlap=CHUNK_OVERLAP)

# -------- Helpers ----------------------------------------------------------
def _clean(text: str) -> str:
    """Limpia referencias [1], tablas y exceso de espacios."""
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _page_titles_from_category(cat: str, depth: int = 1) -> Set[str]:
    """Recorre categoría y (opcional) sub-categorías hasta `depth`."""
    titles, queue = set(), [(cat, 0)]
    while queue:
        current, lvl = queue.pop()
        try:
            cat_page = wikipedia.page(current, auto_suggest=False)
            for link in cat_page.links:
                if link.startswith("Categoría:") and lvl < depth:
                    queue.append((link, lvl + 1))
                else:
                    titles.add(link)
        except Exception:
            continue
    return titles

# -------- Pipeline ---------------------------------------------------------
def ingest(title: str):
    try:
        page = wikipedia.page(title, auto_suggest=False)
    except Exception:
        return False

    chunks = splitter.split_text(_clean(page.content))
    now    = datetime.datetime.utcnow().isoformat()

    col.add(
        documents = chunks,
        ids       = [f"wiki_{page.pageid}_c{i:03d}" for i in range(len(chunks))],
        metadatas = [{
            "source_url"  : page.url,
            "source_title": page.title,
            "created_at"  : now,
            "source_section": "Wikipedia",
        } for _ in chunks],
    )
    print(f"✓ {title:60s} → {len(chunks)} chunks")
    return True


if __name__ == "__main__":
    t0 = time.time()
    cat_titles = _page_titles_from_category("Categoría:Colombia", depth=2)

    # Ingresa ~1 500-2 000 páginas (puedes filtrar si es demasiado)
    for t in sorted(cat_titles):
        ingest(t)

    print(f"\nIngesta masiva terminada en {time.time() - t0:,.1f} s")
