def make_json_serializable(payload):
    for key, value in payload.items():
        try:
            if isinstance(value, dict):
                # recursively sanitize dicts
                payload[key] = make_json_serializable(value.copy())
            elif not isinstance(value, (str, int, float, bool, type(None))):
                # everything else becomes a string
                payload[key] = str(value)
        except Exception:
            # non blocking if it can't cast to a str
            pass
    return payload
