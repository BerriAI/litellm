def normalize_fal_model_id(model: str) -> str:
    stripped = model
    if stripped.startswith("fal_ai/"):
        stripped = stripped[len("fal_ai/") :]
    stripped = stripped.strip("/")
    if not stripped:
        raise ValueError("fal.ai model id is empty after stripping provider prefix")
    return stripped
