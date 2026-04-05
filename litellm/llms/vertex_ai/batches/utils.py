from typing import List, Optional

from litellm.types.llms.openai import Batch


async def get_vertex_ai_batch_output_with_custom_id(
    batch: Batch,
    litellm_params: Optional[dict] = None,
) -> List[dict]:
    """
    Vertex AI batch outputs do not reliably include custom_id.
    Map output lines back to custom_id using keyField (preferred).
    """
    from litellm.files.main import afile_content

    if batch.output_file_id is None:
        raise ValueError("Output file id is None cannot retrieve file content")

    from litellm.batches.batch_utils import (
        _extract_file_access_credentials,
        _get_file_content_as_dictionary,
    )

    credentials = _extract_file_access_credentials(litellm_params)

    output_content = await afile_content(
        file_id=batch.output_file_id,
        custom_llm_provider="vertex_ai",
        **credentials,
    )
    output_lines = _get_file_content_as_dictionary(output_content.content)

    output_has_keys = all(
        (line.get("custom_id") or line.get("key")) for line in output_lines
    )
    if not output_has_keys:
        raise ValueError(
            "Vertex AI batch output is missing custom_id/key; cannot safely map results"
        )

    mapped_lines = []
    for line in output_lines:
        key = line.get("custom_id") or line.get("key")
        if key is not None:
            line["custom_id"] = key
        mapped_lines.append(line)
    return mapped_lines
