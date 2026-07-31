import os
import gc
import logging
from PyPDF2 import PdfReader
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
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text[:2000],
            output_dimensionality=768,
        )
        return result["embedding"]
    except Exception as e:
        logger.error(f"Error generando embedding: {e}", exc_info=True)
        return None


def chunk_text(text, chunk_size=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        last_period = chunk.rfind(".")
        if last_period > chunk_size * 0.3:
            chunk = chunk[: last_period + 1]
            end = start + last_period + 1
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 100]


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

    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            total_pages = len(reader.pages)
            logger.info(f"PDF tiene {total_pages} páginas")

            chunk_count = 0
            embedding_errors = 0
            batch_vectors = []
            batch_size = 20

            for page_num in range(total_pages):
                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if not page_text or len(page_text.strip()) < 50:
                        continue

                    chunks = chunk_text(page_text)
                    logger.info(f"Página {page_num}: {len(chunks)} chunks")

                    for i, chunk in enumerate(chunks):
                        embedding = get_embedding(chunk)
                        if embedding is None:
                            embedding_errors += 1
                            continue

                        batch_vectors.append(
                            {
                                "id": f"{source_name}_p{page_num}_{i}",
                                "values": embedding,
                                "metadata": {
                                    "source": source_name,
                                    "page": page_num,
                                    "chunk_index": chunk_count,
                                    "text": chunk[:800],
                                },
                            }
                        )
                        chunk_count += 1

                        if len(batch_vectors) >= batch_size:
                            index.upsert(vectors=batch_vectors)
                            batch_vectors = []
                            gc.collect()

                except Exception as e:
                    logger.warning(f"Error página {page_num}: {e}")
                    continue

            if batch_vectors:
                index.upsert(vectors=batch_vectors)

            gc.collect()

            logger.info(f"Completado: {chunk_count} chunks, {embedding_errors} errores")

            if chunk_count == 0:
                return 0, "No se pudieron generar embeddings. Verifique la API key."

            return (
                chunk_count,
                f"PDF '{source_name}': {chunk_count} fragmentos indexados.",
            )

    except Exception as e:
        logger.error(f"Error procesando PDF: {e}", exc_info=True)
        return 0, f"Error al leer el PDF: {str(e)}"


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
        scan = index.query(vector=[0.0] * DIMENSION, top_k=10000, include_metadata=True)
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
