from litellm.llms.perplexity.search.transformation import PerplexitySearchConfig


class TestPerplexitySearchRequestTransformation:
    def test_forwards_full_documented_param_set(self):
        config = PerplexitySearchConfig()
        optional_params = {
            "search_after_date_filter": "01/01/2026",
            "search_before_date_filter": "06/15/2026",
            "last_updated_after_filter": "03/01/2026",
            "last_updated_before_filter": "03/31/2026",
            "search_recency_filter": "month",
            "search_language_filter": ["en"],
            "search_context_size": "high",
            "max_tokens": 2048,
            "max_results": 5,
            "search_domain_filter": ["europa.eu"],
            "max_tokens_per_page": 1024,
            "country": "US",
        }

        body = config.transform_search_request(
            query="EU AI Act", optional_params=optional_params
        )

        assert body == {"query": "EU AI Act", **optional_params}

    def test_omits_unset_and_none_optional_params(self):
        config = PerplexitySearchConfig()

        body = config.transform_search_request(
            query="hello",
            optional_params={"search_recency_filter": None, "max_results": 5},
        )

        assert body == {"query": "hello", "max_results": 5}

    def test_passes_through_arbitrary_params(self):
        config = PerplexitySearchConfig()

        body = config.transform_search_request(
            query="hello",
            optional_params={"some_new_perplexity_param": "x", "country": "GB"},
        )

        assert body == {
            "query": "hello",
            "some_new_perplexity_param": "x",
            "country": "GB",
        }

    def test_query_argument_is_not_overridden_by_optional_params(self):
        config = PerplexitySearchConfig()

        body = config.transform_search_request(
            query="real query",
            optional_params={"query": "injected", "search_recency_filter": "week"},
        )

        assert body == {"query": "real query", "search_recency_filter": "week"}

    def test_query_only_request(self):
        config = PerplexitySearchConfig()

        body = config.transform_search_request(query=["a", "b"], optional_params={})

        assert body == {"query": ["a", "b"]}
