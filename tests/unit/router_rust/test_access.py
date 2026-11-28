from typing import Final

import litellm
from litellm.router_rust.steps.access import pinned_deployment_wildcard_grant


def test_pinned_deployment_wildcard_grant_rejects_mismatched_requested_model() -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "shared",
                "litellm_params": {"model": "openai/gpt-6-astra"},
                "model_info": {"id": "shared-pub", "access_groups": ["public"]},
            },
            {
                "model_name": "shared",
                "litellm_params": {"model": "openai/gpt-6-astra"},
                "model_info": {"id": "shared-priv", "access_groups": ["tools-private"]},
            },
            {
                "model_name": "tools-group",
                "litellm_params": {"model": "openai/gpt-6-astra"},
                "model_info": {"id": "tools-private"},
            },
        ]
    )
    try:
        assert (
            pinned_deployment_wildcard_grant(
                "tools-private",
                ("public", "tools*"),
                None,
                router.registry(),
                by_pinned_id=True,
                requested_model="shared",
            )
            is None
        )
    finally:
        router.retire()


def test_pinned_deployment_wildcard_grant_scopes_addition_to_requested_target() -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "alpha",
                "litellm_params": {"model": "openai/gpt-6-astra"},
                "model_info": {"id": "beta"},
            },
            {
                "model_name": "alpha",
                "litellm_params": {"model": "openai/gpt-6-astra"},
                "model_info": {"id": "alpha-ag", "access_groups": ["beta"]},
            },
        ]
    )
    try:
        registry: Final = router.registry()
        assert pinned_deployment_wildcard_grant(
            "beta",
            ("alpha*",),
            None,
            registry,
            by_pinned_id=True,
            requested_model="alpha",
        ) == ("alpha*", "alpha")
        assert pinned_deployment_wildcard_grant(
            "beta",
            ("alpha*",),
            None,
            registry,
            by_pinned_id=True,
            requested_model="beta",
        ) == ("alpha*", "beta")
        assert pinned_deployment_wildcard_grant(
            "beta",
            ("alpha*", "beta"),
            None,
            registry,
            by_pinned_id=True,
            requested_model="alpha",
        ) == ("alpha*", "beta", "alpha")
        assert pinned_deployment_wildcard_grant(
            "beta",
            ("alpha*",),
            None,
            registry,
        ) == ("alpha*", "beta")
    finally:
        router.retire()
