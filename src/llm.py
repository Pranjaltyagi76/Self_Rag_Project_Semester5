"""LLM wrapper for the Self-RAG project.

A single small interface used by both the generator and the critic. The backend
is selected by the LLM_BACKEND environment variable (default: "groq").

Usage:
    from llm import generate
    text = generate("Say hello.", system="You are terse.", max_tokens=10)
"""
import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_BACKEND = os.getenv("LLM_BACKEND", "groq").lower()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")


class LLM:
    """Thin chat wrapper around a hosted or local model."""

    def __init__(self, backend=None, model=None, temperature=0.0):
        self.backend = (backend or DEFAULT_BACKEND).lower()
        self.temperature = temperature

        if self.backend == "groq":
            self.model = model or GROQ_MODEL
            self._client = self._init_groq()
        elif self.backend == "ollama":
            # Offline fallback: a local model served by Ollama. No API key needed.
            self.model = model or OLLAMA_MODEL
            self._client = None
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend!r}")

    def _init_groq(self):
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_groq_api_key_here":
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
                "free key from https://console.groq.com/keys"
            )
        return Groq(api_key=api_key)

    def generate(self, prompt, system=None, temperature=None, max_tokens=512):
        """Return the model's text reply to `prompt` (optionally with a system prompt)."""
        temp = self.temperature if temperature is None else temperature
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if self.backend == "groq":
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()

        if self.backend == "ollama":
            import ollama

            resp = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temp, "num_predict": max_tokens},
            )
            return resp["message"]["content"].strip()

        raise ValueError(f"Unknown LLM backend: {self.backend!r}")


_default_llm = None


def get_llm():
    """Return a lazily-created default LLM configured from the environment."""
    global _default_llm
    if _default_llm is None:
        _default_llm = LLM()
    return _default_llm


def generate(prompt, **kwargs):
    """Convenience wrapper around the default LLM's generate()."""
    return get_llm().generate(prompt, **kwargs)


if __name__ == "__main__":
    print(generate("Reply with exactly the word: OK", max_tokens=5))
