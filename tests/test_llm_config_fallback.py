from document_agent.llm import OpenAIChatLLM


def test_llm_uses_config_fallback_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured: dict[str, str] = {}

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            captured["api_key"] = api_key

    monkeypatch.setattr("document_agent.llm.OpenAI", FakeClient)
    OpenAIChatLLM(model="gpt-4.1-mini")

    assert captured["api_key"].startswith("sk-")

