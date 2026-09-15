from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_relevant_question():
    response = client.post(
        "/ask",
        json={
            "question": "How can I monitor Amazon ECS applications?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0


def test_irrelevant_question():
    response = client.post(
        "/ask",
        json={
            "question": "What is the recipe for making chocolate cake?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sources"] == []
    assert "could not find" in data["answer"].lower()


def test_empty_question():
    response = client.post(
        "/ask",
        json={
            "question": ""
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Question cannot be empty"
    }