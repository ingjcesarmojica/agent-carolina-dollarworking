import os
import chromadb
from chromadb.utils import embedding_functions
import pdfplumber


CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
PDF_DIR = os.path.join(os.path.dirname(__file__), "knowledge_pdfs")


def init_dirs():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)


def get_collection():
    init_dirs()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name="tusabogados_laboral",
        embedding_function=ef
    )
    return collection


def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        last_period = chunk.rfind(".")
        if last_period > chunk_size * 0.5:
            chunk = chunk[:last_period + 1]
            end = start + last_period + 1
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if len(c) > 50]


def add_pdf(pdf_path, source_name=None):
    collection = get_collection()
    if source_name is None:
        source_name = os.path.basename(pdf_path)
    existing = collection.get(where={"source": source_name})
    if existing and existing["ids"]:
        collection.delete(ids=existing["ids"])
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return 0, "No se pudo extraer texto del PDF."
    chunks = chunk_text(text)
    if not chunks:
        return 0, "El PDF no contiene texto procesable."
    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    return len(chunks), f"PDF '{source_name}' procesado: {len(chunks)} fragmentos indexados."


def search_knowledge(query, n_results=3):
    collection = get_collection()
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query], n_results=n_results)
    docs = []
    if results and results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            docs.append({
                "text": doc,
                "source": meta.get("source", "desconocido")
            })
    return docs


def list_documents():
    collection = get_collection()
    if collection.count() == 0:
        return []
    all_data = collection.get()
    sources = set()
    if all_data and all_data["metadatas"]:
        for meta in all_data["metadatas"]:
            sources.add(meta.get("source", "desconocido"))
    return list(sources)


def delete_document(source_name):
    collection = get_collection()
    existing = collection.get(where={"source": source_name})
    if existing and existing["ids"]:
        collection.delete(ids=existing["ids"])
        return True, f"Documento '{source_name}' eliminado."
    return False, "Documento no encontrado."
