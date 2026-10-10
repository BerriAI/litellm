from litellm.llms.perplexity.search.transformation import PerplexitySearchConfig


class TestPerplexitySearchRequestTransformation:
    def setup_method(self):
        self.config = PerplexitySearchConfig()

    def test_forwards_full_documented_param_set(self):
        optional_params = {
            "search_after_date_filter": "01/01/2026",
            "search_before_date_filter": "06/15/2026",
            "last_updated_after_filter": "03/01/2026",
            "last_updated_before_filter": "03/31/2026",
            "search_recency_filter": "month",
            "search_language_filter": ["en"],
            "search_context_size": "high",
            "search_mode": "web",
            "max_tokens": 2048,
            "max_results": 5,
            "search_domain_filter": ["europa.eu"],
            "max_tokens_per_page": 1024,
            "country": "US",
        }

        body = self.config.transform_search_request(query="EU AI Act", optional_params=optional_params)

        assert body == {"query": "EU AI Act", **optional_params}

    def test_omits_unset_and_none_optional_params(self):
        body = self.config.transform_search_request(
            query="EU AI Act",
            optional_params={
                "max_results": 5,
                "search_recency_filter": None,
                "search_language_filter": None,
                "unknown_param": "ignored",
            },
        )

        assert body == {"query": "EU AI Act", "max_results": 5, "unknown_param": "ignored"}

    def test_query_is_not_overwritten_by_optional_params(self):
        body = self.config.transform_search_request(
            query="original query",
            optional_params={"query": "injected", "max_results": 3},
        )

        assert body["query"] == "original query"
        assert body["max_results"] == 3


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
            "search_mode": "web",
            "max_tokens": 2048,
            "max_results": 5,
            "search_domain_filter": ["europa.eu"],
            "max_tokens_per_page": 1024,
            "country": "US",
        }

        body = config.transform_search_request(query="EU AI Act", optional_params=optional_params)

        assert body == {"query": "EU AI Act", **optional_params}

    def test_omits_unset_and_none_optional_params(self):
        config = PerplexitySearchConfig()
        body = config.transform_search_request(
            query="EU AI Act",
            optional_params={
                "max_results": 5,
                "search_recency_filter": None,
                "search_language_filter": None,
                "unknown_param": "ignored",
            },
        )

        assert body == {"query": "EU AI Act", "max_results": 5, "unknown_param": "ignored"}

    def test_query_is_not_overwritten_by_optional_params(self):
        config = PerplexitySearchConfig()
        body = config.transform_search_request(
            query="original query",
            optional_params={"query": "injected", "max_results": 3},
        )

        assert body["query"] == "original query"
        assert body["max_results"] == 3
