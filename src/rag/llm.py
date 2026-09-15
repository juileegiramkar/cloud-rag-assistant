import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


class LocalLLM:
    """
    Wrapper around a locally running Ollama LLM.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the local LLM.
        """

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )

        response.raise_for_status()

        return response.json()["response"]