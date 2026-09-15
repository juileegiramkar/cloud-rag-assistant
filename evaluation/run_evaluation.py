import json

from src.ingestion.embedder import Embedder
from src.rag.vector_store import VectorStore


# Load evaluation dataset.
with open(
    "evaluation/eval_dataset.json",
    "r",
    encoding="utf-8",
) as file:
    dataset = json.load(file)


# Load RAG retrieval components.
embedder = Embedder()

vector_store = VectorStore.load(
    "data/processed/vector_store"
)


hit_at_1 = []
hit_at_3 = []
hit_at_5 = []
reciprocal_ranks = []


for item in dataset:
    question = item["question"]
    relevant_pages = set(item["relevant_pages"])

    # Out-of-scope question.
    if not relevant_pages:
        continue

    query_embedding = embedder.embed_query(question)

    results = vector_store.search(
        query_embedding,
        top_k=5,
    )

    retrieved_pages = [
        result["page"]
        for result in results
    ]

    # Hit@1
    hit_at_1.append(
        int(
            any(
                page in relevant_pages
                for page in retrieved_pages[:1]
            )
        )
    )

    # Hit@3
    hit_at_3.append(
        int(
            any(
                page in relevant_pages
                for page in retrieved_pages[:3]
            )
        )
    )

    # Hit@5
    hit_at_5.append(
        int(
            any(
                page in relevant_pages
                for page in retrieved_pages[:5]
            )
        )
    )

    # Reciprocal Rank.
    rank = 0

    for position, page in enumerate(
        retrieved_pages,
        start=1,
    ):
        if page in relevant_pages:
            rank = position
            break

    if rank:
        reciprocal_ranks.append(1 / rank)
    else:
        reciprocal_ranks.append(0)


def average(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


print("\n" + "=" * 60)
print("RAG RETRIEVAL EVALUATION")
print("=" * 60)

print(f"\nEvaluation questions: {len(hit_at_1)}")

print(
    f"Hit@1: {average(hit_at_1):.2%}"
)

print(
    f"Hit@3: {average(hit_at_3):.2%}"
)

print(
    f"Hit@5: {average(hit_at_5):.2%}"
)

print(
    f"MRR:    {average(reciprocal_ranks):.3f}"
)

print("\nPer-question results:")

for item in dataset:
    question = item["question"]
    relevant_pages = set(item["relevant_pages"])

    if not relevant_pages:
        continue

    query_embedding = embedder.embed_query(question)

    results = vector_store.search(
        query_embedding,
        top_k=5,
    )

    retrieved_pages = [
        result["page"]
        for result in results
    ]

    first_hit = None

    for position, page in enumerate(
        retrieved_pages,
        start=1,
    ):
        if page in relevant_pages:
            first_hit = position
            break

    print(
        f"\n{question}"
    )
    print(
        f"Retrieved: {retrieved_pages}"
    )
    print(
        f"First relevant result: "
        f"{first_hit if first_hit else 'None'}"
    )