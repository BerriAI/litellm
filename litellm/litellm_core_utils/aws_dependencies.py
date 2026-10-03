from importlib import import_module


def require_aws_sdk() -> None:
    try:
        import_module("boto3")
        import_module("botocore")
    except ModuleNotFoundError as error:
        if error.name not in ("boto3", "botocore"):
            raise
        raise ImportError('Install AWS support with pip install "litellm[aws]"') from error
