from __future__ import annotations

from pathlib import Path


def tiktoken_resource_dir() -> str:
    from litellm.litellm_core_utils import tokenizers

    return str(Path(tokenizers.__file__).parent)


def download_tokenizer(repo_id: str, revision: str, token: str | None) -> str:
    import huggingface_hub

    return huggingface_hub.hf_hub_download(  # pyright: ignore[reportUnknownMemberType]  # untyped user_agent dict in its signature
        repo_id=repo_id, filename="tokenizer.json", revision=revision, token=token
    )
