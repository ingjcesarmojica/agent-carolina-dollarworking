import os
import logging
import pdfplumber
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

logger = logging.getLogger(__name__)


INDEX_NAME = "tusabogados-laboral"
DIMENSION = 768


def get_pc():
    api_key = os.environ.get("PINECONE_API_KEY", "")
    if not api_key:
        logger.error("PINECONE_API_KEY no configurada")
        return None
    try:
        pc = Pinecone(api_key=api_key)
        logger.info("Pinecone conectado exitosamente")
        return pc
    except Exception as e:
        logger.error(f"Error conectando a Pinecone: {e}")
        return None


def get_index():
    pc = get_pc()
    if pc is None:
        return None
    try:
        existing = pc.list_indexes()
        index_names = [idx.name for idx in existing.indexes]
        logger.info(f"Índices existentes: {index_names}")
        if INDEX_NAME not in index_names:
            logger.info(f"Creando índice {INDEX_NAME}...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info(f"Índice {INDEX_NAME} creado")
        return pc.Index(INDEX_NAME)
    except Exception as e:
        logger.error(f"Error obteniendo índice: {e}")
        return None


def get_embedding(text):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.error("GEMINI_API_KEY no configurada")
        return None
    try:
        genai.configure(api_key=api_key)
        result = genai.embed_content(model="models/text-embedding-004", content=text)
        return result["embedding"]
    except Exception as e:
        logger.error(f"Error generando embedding: {e}")
        return None


def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        last_period = chunk.rfind(".")
        if last_period > chunk_size * 0.4:
            chunk = chunk[: last_period + 1]
            end = start + last_period + 1
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 50]


def add_pdf(pdf_path, source_name=None):
    index = get_index()
    if index is None:
        return 0, "Pinecone no configurado. Verifique PINECONE_API_KEY."

    if source_name is None:
        source_name = os.path.basename(pdf_path)

    try:
        index.delete(filter={"source": source_name})
    except Exception:
        pass

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return 0, "No se pudo extraer texto del PDF."

    chunks = chunk_text(text)
    if not chunks:
        return 0, "El PDF no contiene texto procesable."

    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if embedding is None:
            continue
        vectors.append(
            {
                "id": f"{source_name}_{i}",
                "values": embedding,
                "metadata": {
                    "source": source_name,
                    "chunk_index": i,
                    "text": chunk[:1000],
                },
            }
        )

    if not vectors:
        return 0, "No se pudieron generar embeddings para el PDF."

    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)

    return len(
        vectors
    ), f"PDF '{source_name}' procesado: {len(vectors)} fragmentos indexados."


def search_knowledge(query, n_results=3):
    index = get_index()
    if index is None:
        return []

    query_embedding = get_embedding(query)
    if query_embedding is None:
        return []

    try:
        stats = index.describe_index_stats()
        if stats.total_vector_count == 0:
            return []
    except Exception:
        return []

    results = index.query(
        vector=query_embedding, top_k=n_results, include_metadata=True
    )

    docs = []
    for match in results.matches:
        metadata = match.metadata
        docs.append(
            {
                "text": metadata.get("text", ""),
                "source": metadata.get("source", "desconocido"),
            }
        )
    return docs


def list_documents():
    index = get_index()
    if index is None:
        return []

    try:
        stats = index.describe_index_stats()
        if stats.total_vector_count == 0:
            return []
    except Exception:
        return []

    sources = set()
    try:
        namespaces = stats.namespaces
        for ns_name in namespaces:
            ns_stats = index.describe_index_stats()
            break
    except Exception:
        pass

    try:
        scan = index.query(vector=[0] * DIMENSION, top_k=10000, include_metadata=True)
        for match in scan.matches:
            sources.add(match.metadata.get("source", "desconocido"))
    except Exception:
        pass

    return list(sources)


def delete_document(source_name):
    index = get_index()
    if index is None:
        return False, "Pinecone no configurado."

    try:
        index.delete(filter={"source": source_name})
        return True, f"Documento '{source_name}' eliminado."
    except Exception as e:
        return False, f"Error al eliminar: {str(e)}"
