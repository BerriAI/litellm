from pydantic import ValidationError


def stringify_pydantic_error(validation_error: ValidationError) -> str:
    return "\n".join(
        f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
        for e in validation_error.errors()
    )
