import os

from langchain_google_genai import ChatGoogleGenerativeAI

_llm: ChatGoogleGenerativeAI | None = None


def get_llm() -> ChatGoogleGenerativeAI:
    """Return a shared ChatGoogleGenerativeAI instance, initialized lazily."""
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
    return _llm