import base64
from litellm.litellm_core_utils.prompt_templates.factory import convert_to_gemini_tool_call_result

# Minimal 1x1 red PNG as base64
TINY_PNG_B64 = base64.b64encode(
    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
    b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00'
    b'\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05'
    b'\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
).decode()

# Case 1: Anthropic-native image block in list content
msg_list = {
    "role": "tool",
    "tool_call_id": "tool_1",
    "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": TINY_PNG_B64}},
    ],
}

# Case 2: data-URL string (serialised by transformation.py)
msg_dataurl = {
    "role": "tool",
    "tool_call_id": "tool_1",
    "content": f"data:image/png;base64,{TINY_PNG_B64}",
}

last_message = {
    "role": "assistant",
    "tool_calls": [
        {"id": "tool_1", "function": {"name": "read_file"}, "type": "function"}
    ],
}

for label, msg in [("list content", msg_list), ("data-URL string", msg_dataurl)]:
    result = convert_to_gemini_tool_call_result(msg, last_message_with_tool_calls=last_message)
    parts = result if isinstance(result, list) else [result]
    has_inline = any(
        (p.get("inline_data") if isinstance(p, dict) else getattr(p, "inline_data", None))
        for p in parts
    )
    print(f"{label}: inline_data present = {has_inline}")
    if not has_inline:
        print(f"  FAIL — image was dropped: {parts}")
