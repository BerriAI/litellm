from typing import Final

from typing_extensions import assert_type

import litellm


def constructed_router() -> None:
    router: Final = litellm.Router(
        model_list=[{"model_name": "gpt", "litellm_params": {"model": "openai/gpt-4o"}}],
        num_retries=2,
    )
    assert_type(router, litellm.Router)


def misspelled_parameter() -> None:
    litellm.Router(model_lst=[])  # pyright: ignore[reportCallIssue]  # Router(...) refuses a name it does not take
