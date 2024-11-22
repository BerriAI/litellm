import google.generativeai as genai
import pytest

genai.configure(
    api_key="sk-1234",
    client_options={"api_endpoint": "http://0.0.0.0:4000/gemini"},
    transport="rest",
)


def test_basic_non_streaming():
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Explain how AI works")
    print("response", response)
    assert response.text is not None
