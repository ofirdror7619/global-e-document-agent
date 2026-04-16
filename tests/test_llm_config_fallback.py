from document_agent.llm import GeminiChatLLM


def test_llm_uses_config_fallback_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    captured: dict[str, str] = {}

    class FakeSession:
        def post(self, *args, **kwargs):
            _ = (args, kwargs)
            raise AssertionError("post should not be called in this test")

    monkeypatch.setattr(
        GeminiChatLLM,
        "_read_api_key_from_config",
        staticmethod(lambda: "AIzaFakeConfigKey123"),
    )
    monkeypatch.setattr("document_agent.llm.requests.Session", lambda: FakeSession())

    llm = GeminiChatLLM(model="gemini-2.5-flash-lite")
    captured["api_key"] = llm._api_key

    assert captured["api_key"].startswith("AIza")
