from typing import List

class OpenrouterConfig():
    """
    Reference: https://openrouter.ai/docs#format

    """
    # OpenRouter-only parameters
    transforms: List[str] = []
    models: List[str] = []
    route: str = ''
