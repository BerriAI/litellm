import base64
import binascii
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, NoReturn, Optional, Tuple

from litellm.constants import request_timeout

REDUCTO_API_BASE = "https://platform.reducto.ai"
REDUCTO_ID_PREFIX = "reducto://"

if TYPE_CHECKING:
    from litellm.llms.base_llm.ocr.transformation import OCRPage


def _normalize_api_base(api_base: Optional[str]) -> str:
    return (api_base or REDUCTO_API_BASE).rstrip("/")


def _raise_bad_request(message: str, model: str) -> NoReturn:
    import litellm

    raise litellm.BadRequestError(
        message=message,
        model=model,
        llm_provider="reducto",
    )


def extract_file_id_or_bytes(
    source_url: str,
    model: str,
) -> Tuple[Optional[str], Optional[bytes], Optional[str]]:
    if source_url.startswith(REDUCTO_ID_PREFIX):
        return source_url, None, None

    if source_url.startswith("http://") or source_url.startswith("https://"):
        _raise_bad_request(
            "Reducto requires type='file' (auto-uploaded) or a reducto:// id. Plain http(s) URLs are not supported; upload the file first.",
            model=model,
        )

    if not source_url.startswith("data:"):
        _raise_bad_request(
            "Reducto requires a reducto:// id or a base64 data URI after OCR preprocessing.",
            model=model,
        )

    try:
        header, encoded = source_url.split(",", 1)
    except ValueError:
        _raise_bad_request("Invalid Reducto data URI provided.", model=model)

    if ";base64" not in header:
        _raise_bad_request(
            "Reducto only supports base64-encoded data URIs.", model=model
        )

    mime = header.removeprefix("data:").split(";")[0] or "application/octet-stream"
    try:
        raw_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        _raise_bad_request("Invalid Reducto base64 payload provided.", model=model)

    return None, raw_bytes, mime


def upload_bytes_sync(
    raw_bytes: bytes,
    mime: Optional[str],
    api_key: str,
    api_base: Optional[str],
) -> str:
    import litellm

    response = litellm.module_level_client.post(
        url="{}{}".format(_normalize_api_base(api_base), "/upload"),
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("document", raw_bytes, mime or "application/octet-stream")},
        timeout=request_timeout,
    )
    response.raise_for_status()
    return response.json()["file_id"]


async def upload_bytes_async(
    raw_bytes: bytes,
    mime: Optional[str],
    api_key: str,
    api_base: Optional[str],
) -> str:
    import litellm

    response = await litellm.module_level_aclient.post(
        url="{}{}".format(_normalize_api_base(api_base), "/upload"),
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("document", raw_bytes, mime or "application/octet-stream")},
        timeout=request_timeout,
    )
    response.raise_for_status()
    return response.json()["file_id"]


def build_pages_from_reducto(result: Dict[str, Any]) -> List["OCRPage"]:
    from litellm.llms.base_llm.ocr.transformation import OCRPage

    chunks = result.get("chunks", []) or []
    blocks_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for chunk in chunks:
        for block in chunk.get("blocks", []) or []:
            page_no = (block.get("bbox") or {}).get("page")
            if page_no is None:
                continue
            try:
                normalized_page = int(page_no)
            except (TypeError, ValueError):
                continue
            blocks_by_page[normalized_page].append(block)

    if not blocks_by_page:
        fallback_markdown = "\n\n".join(
            chunk.get("content", "") for chunk in chunks if chunk.get("content")
        )
        if fallback_markdown == "":
            return []
        return [OCRPage(index=0, markdown=fallback_markdown)]

    pages: List["OCRPage"] = []
    for page_no, blocks in sorted(blocks_by_page.items()):
        markdown = "\n\n".join(
            block.get("content", "") for block in blocks if block.get("content")
        )
        page_index = max(page_no - 1, 0)
        page = OCRPage(
            index=page_index,
            markdown=markdown,
        )
        # OCRPage accepts extra keys at runtime; assign blocks after construction
        # so static typing does not reject provider-specific metadata.
        setattr(page, "blocks", blocks)
        pages.append(page)
    return pages
