from fastapi import FastAPI
from pydantic import BaseModel

from src.ingestion.embedder import Embedder
from src.rag.vector_store import VectorStore
from src.rag.llm import LocalLLM
from src.rag.rag_pipeline import RAGPipeline


app = FastAPI(
    title="Cloud RAG Assistant",
    description="AI knowledge assistant for Amazon ECS documentation",
    version="1.0.0",
)


# Load RAG components when the application starts.
embedder = Embedder()

vector_store = VectorStore.load(
    "data/processed/vector_store"
)

llm = LocalLLM()

rag = RAGPipeline(
    embedder=embedder,
    vector_store=vector_store,
    llm=llm,
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Cloud RAG Assistant API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    result = rag.answer(
        request.question
    )

    return result