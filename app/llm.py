"""Chat model factory. The rest of the code only sees LangChain's common chat-model interface,
so switching between Gemini and OpenAI is a one-line config change."""

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings


def get_llm(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # timeout + max_retries: if Gemini is slow or unreachable, fail with a visible error
        # after ~60 s instead of leaving the chat hanging forever.
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model, temperature=0, timeout=60, max_retries=2
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=settings.openai_model, temperature=0, timeout=60, max_retries=2)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
