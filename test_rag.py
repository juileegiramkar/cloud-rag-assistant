from src.ingestion.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.llm import LocalLLM
from src.rag.rag_pipeline import RAGPipeline


# Load the existing components
embedder = Embedder()

vector_store = VectorStore.load(
    "data/processed/vector_store"
)

llm = LocalLLM()


# Create the RAG pipeline
rag = RAGPipeline(
    embedder=embedder,
    vector_store=vector_store,
    llm=llm,
)


# Ask a question
question = "How can I monitor Amazon ECS applications?"

result = rag.answer(
    question,
    top_k=5,
)


# Display the answer
print("\nANSWER:")
print(result["answer"])


# Display the sources
print("\nSOURCES:")

for source in result["sources"]:
    print(
        f"Page {source['page']} | "
        f"Score: {source['score']:.4f} | "
        f"{source['chunk_id']}"
    )