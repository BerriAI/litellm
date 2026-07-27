def parse_cache_control_header(header_value: str | None) -> dict[str, int | bool | str]:
    if not header_value or not isinstance(header_value, str):
        return {}

    directives = [d.strip() for d in header_value.split(",") if d.strip()]
    if not directives:
        return {}

    parsed: dict[str, int | bool | str] = {}
    for directive in directives:
        if "=" in directive:
            parts = directive.split("=", 1)
            raw_key = parts[0].strip().lower()
            raw_val = parts[1].strip().strip('"')
            if raw_key in ("max-age", "s-maxage", "s-max-age", "ttl"):
                try:
                    int_val = int(raw_val)
                    parsed["s-maxage"] = int_val
                    parsed["s-max-age"] = int_val
                    parsed["ttl"] = int_val
                except ValueError:
                    parsed[raw_key] = raw_val
            else:
                parsed[raw_key] = raw_val
        else:
            raw_key = directive.lower()
            parsed[raw_key] = True

    return parsed
