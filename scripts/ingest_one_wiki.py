"""
Ingesta rápida de un único artículo de Wikipedia en la colección
`colombia_documents` de ChromaDB.

Ejemplo de ejecución:
    python scripts/ingest_one_wiki.py "Peso colombiano"
"""

import sys, uuid, datetime
import wikipedia                                   # pip install wikipedia
import chromadb                                    # pip install chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
from langchain.text_splitter import RecursiveCharacterTextSplitter  # pip install langchain-text-splitters

# ----- Parámetros básicos ---------------------------------------------------
PAGE          = sys.argv[1] if len(sys.argv) > 1 else "Colombia"
COLLECTION    = "colombia_documents"
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR   = "./data/vectorstore"
# ----------------------------------------------------------------------------

# 1) Descarga el artículo
wikipedia.set_lang("es")
page = wikipedia.page(PAGE, auto_suggest=False)
print(f"Descargado: {page.title}  ->  {page.url}")

# 2) Divide en chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_text(page.content)
print(f"{len(chunks)} chunks generados (size {CHUNK_SIZE}, overlap {CHUNK_OVERLAP})")

# 3) Embeddings
model = SentenceTransformer(MODEL_NAME)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=MODEL_NAME)
embeddings = emb_fn(chunks)   # lista de arrays

# 4) Conecta (o crea) la colección en Chroma
client = chromadb.PersistentClient(path=PERSIST_DIR)
col    = client.get_or_create_collection(COLLECTION)

# 5) Inserta documentos
doc_id  = f"{PAGE.replace(' ', '_').lower()}_{uuid.uuid4().hex[:8]}"
now     = datetime.datetime.utcnow().isoformat()

col.add(
    ids        =[f"{doc_id}_chunk_{i:03d}" for i in range(len(chunks))],
    documents  =chunks,
    metadatas  =[{
        "document_id"   : doc_id,
        "source_url"    : page.url,
        "source_section": "Artículo completo",
        "chunk_index"   : i,
        "created_at"    : now,
        "model_name"    : MODEL_NAME
    } for i in range(len(chunks))],
    embeddings =embeddings
)

print(f"✓ Ingestados {len(chunks)} chunks en la colección «{COLLECTION}»")
