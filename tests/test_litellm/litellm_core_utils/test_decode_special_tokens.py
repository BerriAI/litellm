from litellm import decode, encode
from tokenizers import Tokenizer

TOKENIZER_JSON = """{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":3,"content":"[BOS]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"[BOS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"[BOS]":{"id":"[BOS]","ids":[3],"tokens":["[BOS]"]}}},"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"Hello":1,"World":2},"unk_token":"[UNK]"}}"""


def _create_custom_tokenizer():
    tokenizer = Tokenizer.from_str(TOKENIZER_JSON)
    return {"type": "huggingface_tokenizer", "tokenizer": tokenizer}


def test_decode_can_preserve_huggingface_special_tokens():
    custom_tokenizer = _create_custom_tokenizer()
    sample_text = "Hello World"
    tokens = encode(text=sample_text, custom_tokenizer=custom_tokenizer)

    decoded_text = decode(tokens=tokens, custom_tokenizer=custom_tokenizer)
    decoded_text_with_special_tokens = decode(
        tokens=tokens,
        custom_tokenizer=custom_tokenizer,
        skip_special_tokens=False,
    )

    assert decoded_text == sample_text
    assert decoded_text_with_special_tokens == "[BOS] Hello World"
