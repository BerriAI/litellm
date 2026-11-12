from typing import Final


def ensure_optional_import(module: str) -> None:
    try:
        __import__(module)
    except ModuleNotFoundError as error:
        if error.name != module:
            raise
        package: Final = "boto3" if module == "botocore" else module
        raise ModuleNotFoundError(
            f"Missing optional dependency '{module}'. Run 'pip install {package}'.", name=module
        ) from error
