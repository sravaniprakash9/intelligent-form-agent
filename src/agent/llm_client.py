"""Local Ollama LLM client — no cloud APIs."""

from __future__ import annotations

import httpx

from src.config.settings import settings


class OllamaClient:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.ollama_model
        self.timeout = timeout

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.host}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False},
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]

    def is_available(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                r = client.get(f"{self.host}/api/tags")
                return r.status_code == 200
        except httpx.HTTPError:
            return False
