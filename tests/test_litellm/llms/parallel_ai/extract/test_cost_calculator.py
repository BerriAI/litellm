import pytest

from litellm.llms.parallel_ai.extract.cost_calculator import parallel_ai_extract_cost


def test_extract_cost_prefers_provider_usage_over_requested_urls() -> None:
    cost = parallel_ai_extract_cost(
        request_body={"urls": ["https://example.com/1", "https://example.com/2", "https://example.com/3"]},
        response_body={"usage": [{"name": "sku_extract_excerpts", "count": 2}]},
    )

    assert cost == pytest.approx(0.002)


def test_extract_cost_sums_repeated_usage_skus() -> None:
    cost = parallel_ai_extract_cost(
        request_body={"urls": ["https://example.com/1"]},
        response_body={
            "usage": [
                {"name": "sku_extract_excerpts", "count": 1},
                {"name": "unrelated_sku", "count": 9},
                {"name": "sku_extract_excerpts", "count": 2},
            ]
        },
    )

    assert cost == pytest.approx(0.003)


def test_extract_cost_ignores_malformed_unrelated_usage_skus() -> None:
    cost = parallel_ai_extract_cost(
        request_body={"urls": ["https://example.com/1", "https://example.com/2"]},
        response_body={
            "usage": [
                {"name": "sku_extract_excerpts", "count": 1},
                {"name": "unrelated_sku", "count": "not-an-integer"},
            ]
        },
    )

    assert cost == pytest.approx(0.001)


def test_extract_cost_treats_usage_without_extract_sku_as_unbilled() -> None:
    cost = parallel_ai_extract_cost(
        request_body={"urls": ["https://example.com/1", "https://example.com/2"]},
        response_body={"usage": [{"name": "unrelated_sku", "count": 2}]},
    )

    assert cost == 0.0


def test_extract_cost_treats_empty_usage_as_unbilled() -> None:
    cost = parallel_ai_extract_cost(
        request_body={"urls": ["https://example.com/1", "https://example.com/2"]},
        response_body={"usage": []},
    )

    assert cost == 0.0


def test_extract_cost_falls_back_to_requested_url_count_without_usage() -> None:
    cost = parallel_ai_extract_cost(
        request_body={"urls": ["https://example.com/1", "https://example.com/2"]},
        response_body={"results": []},
    )

    assert cost == pytest.approx(0.002)


@pytest.mark.parametrize("invalid_count", [True, -1, "2"])
def test_extract_cost_falls_back_when_provider_usage_is_invalid(invalid_count: object) -> None:
    cost = parallel_ai_extract_cost(
        request_body={"urls": ["https://example.com/1", "https://example.com/2"]},
        response_body={"usage": [{"name": "sku_extract_excerpts", "count": invalid_count}]},
    )

    assert cost == pytest.approx(0.002)


@pytest.mark.parametrize(
    "request_body",
    [
        {},
        {"urls": "https://example.com"},
        {"urls": ["https://example.com", 42]},
    ],
)
def test_extract_cost_does_not_guess_from_invalid_request_urls(request_body: object) -> None:
    assert parallel_ai_extract_cost(request_body=request_body, response_body={}) == 0.0
