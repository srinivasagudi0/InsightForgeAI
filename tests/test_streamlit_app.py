from __future__ import annotations


def test_streamlit_app_imports() -> None:
    import streamlit_app

    assert callable(streamlit_app.main)


def test_resolve_backend_api_key_from_streamlit_secrets() -> None:
    import streamlit_app

    api_key = streamlit_app.resolve_backend_api_key(
        secrets_source={"OPENAI_API_KEY": " secret-key "},
        env={},
    )

    assert api_key == "secret-key"


def test_resolve_backend_api_key_falls_back_to_env() -> None:
    import streamlit_app

    api_key = streamlit_app.resolve_backend_api_key(
        secrets_source={},
        env={"OPENAI_API_KEY": " env-key "},
    )

    assert api_key == "env-key"
