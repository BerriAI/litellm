import pytest

from litellm.router_utils.pattern_match_deployments import PatternMatchRouter


@pytest.fixture
def wildcard_deployment() -> dict:
    return {
        "model_name": "bedrock/*",
        "litellm_params": {
            "model": "bedrock/*",
            "extra_headers": {"x-pool": "pool-1"},
        },
        "model_info": {
            "id": "wildcard-bedrock-1",
            "large_metadata": {f"k{i}": {"v": i} for i in range(100)},
        },
    }


class TestReturnPatternMatchedDeployments:
    def test_rewrites_model_name_without_mutating_source(self, wildcard_deployment):
        pattern_router = PatternMatchRouter()
        pattern_router.add_pattern("bedrock/*", wildcard_deployment)

        result = pattern_router.route("bedrock/anthropic.claude-v2")

        assert result is not None
        assert result[0]["litellm_params"]["model"] == "bedrock/anthropic.claude-v2"
        assert wildcard_deployment["litellm_params"]["model"] == "bedrock/*"

    def test_does_not_deepcopy_model_info(self, wildcard_deployment):
        """Regression test for https://github.com/BerriAI/litellm/issues/33636

        Deployments can carry large model_info blobs (tens of KB of JSON for
        DB-stored models). Deep-copying model_info on every route() call made
        GET /v1/models cost minutes of event-loop CPU under wildcard configs.
        Only litellm_params is rewritten, so model_info must be shared.
        """
        pattern_router = PatternMatchRouter()
        pattern_router.add_pattern("bedrock/*", wildcard_deployment)

        result = pattern_router.route("bedrock/anthropic.claude-v2")

        assert result is not None
        assert result[0]["model_info"] is wildcard_deployment["model_info"]

    def test_returned_litellm_params_are_isolated_from_source(self, wildcard_deployment):
        pattern_router = PatternMatchRouter()
        pattern_router.add_pattern("bedrock/*", wildcard_deployment)

        result = pattern_router.route("bedrock/anthropic.claude-v2")

        assert result is not None
        result[0]["litellm_params"]["api_key"] = "sk-request-scoped"
        result[0]["litellm_params"]["extra_headers"]["x-pool"] = "mutated"
        assert "api_key" not in wildcard_deployment["litellm_params"]
        assert wildcard_deployment["litellm_params"]["extra_headers"]["x-pool"] == "pool-1"

    def test_multiple_deployments_under_one_pattern(self, wildcard_deployment):
        second = {
            "model_name": "bedrock/*",
            "litellm_params": {"model": "bedrock/*"},
            "model_info": {"id": "wildcard-bedrock-2"},
        }
        pattern_router = PatternMatchRouter()
        pattern_router.add_pattern("bedrock/*", wildcard_deployment)
        pattern_router.add_pattern("bedrock/*", second)

        result = pattern_router.route("bedrock/amazon.titan-text-express-v1")

        assert result is not None
        assert len(result) == 2
        assert all(d["litellm_params"]["model"] == "bedrock/amazon.titan-text-express-v1" for d in result)
