from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from litellm import decode, encode


def _create_custom_tokenizer():
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "Hello": 1, "World": 2}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens([AddedToken("[BOS]", special=True)])
    bos_token_id = tokenizer.token_to_id("[BOS]")
    assert bos_token_id is not None
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A",
        special_tokens=[("[BOS]", bos_token_id)],
    )
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
