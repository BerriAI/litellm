from litellm import decode, encode


def test_decode_can_preserve_huggingface_special_tokens():
    sample_text = "Hello World, this is my input string!"
    tokens = encode(model="meta-llama/Llama-2-7b-chat", text=sample_text)

    decoded_text = decode(model="meta-llama/Llama-2-7b-chat", tokens=tokens)
    decoded_text_with_special_tokens = decode(
        model="meta-llama/Llama-2-7b-chat",
        tokens=tokens,
        skip_special_tokens=False,
    )

    assert decoded_text == sample_text
    assert sample_text in decoded_text_with_special_tokens
    assert decoded_text_with_special_tokens != sample_text
