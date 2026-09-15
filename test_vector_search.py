from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import chunk_pages
from src.ingestion.embedder import Embedder
from src.rag.vector_store import VectorStore


PDF_PATH = "data/raw/ecs-best-practices.pdf"


# 1. Load PDF
pages = load_pdf(PDF_PATH)
print("Pages:", len(pages))


# 2. Create chunks
chunks = chunk_pages(pages)
print("Chunks:", len(chunks))


# 3. Create embeddings
embedder = Embedder()

texts = [chunk["text"] for chunk in chunks]

embeddings = embedder.embed_texts(texts)

print("Embeddings:", len(embeddings))
print("Embedding dimensions:", len(embeddings[0]))


# 4. Create FAISS vector store
vector_store = VectorStore(
    dimension=len(embeddings[0])
)


# 5. Add document vectors
vector_store.add(
    embeddings,
    chunks,
)

print("Vectors in FAISS:", vector_store.index.ntotal)

vector_store.save(
    "data/processed/vector_store"
)

print("Vector store saved.")
# 6. Search
query = "How can I monitor Amazon ECS applications?"

query_embedding = embedder.embed_query(query)

results = vector_store.search(
    query_embedding,
    top_k=5,
)


# 7. Display results
print("\nTop results:\n")

for i, result in enumerate(results, start=1):
    print(f"Result {i}")
    print(f"Score: {result['score']:.4f}")
    print(f"Page: {result['page']}")
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Text: {result['text'][:300]}")
    print("-" * 60)