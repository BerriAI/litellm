from litellm.llms.vertex_ai.gemini.grounding_requests import (
    GroundingRequests,
    calculate_grounding_requests,
)


def test_search_only_counts_non_empty_queries_as_web_requests():
    result = calculate_grounding_requests(
        [
            {
                "webSearchQueries": ["", "capital of France", "France capital"],
                "groundingChunks": [{"web": {"uri": "https://example.com", "title": "Example"}}],
            }
        ]
    )
    assert result == GroundingRequests(web_search_requests=2, google_maps_grounding_requests=None)


def test_gemini_api_maps_only_counts_queries_as_maps_requests():
    result = calculate_grounding_requests(
        [
            {
                "webSearchQueries": ["coffee shops near the Louvre"],
                "groundingChunks": [{"maps": {"uri": "https://maps.google.com/?cid=1", "placeId": "p1"}}],
            }
        ]
    )
    assert result == GroundingRequests(web_search_requests=None, google_maps_grounding_requests=1)


def test_vertex_maps_only_without_queries_counts_one_maps_request():
    result = calculate_grounding_requests(
        [
            {
                "groundingChunks": [
                    {"maps": {"uri": "https://maps.google.com/?cid=1", "placeId": "p1"}},
                    {"maps": {"uri": "https://maps.google.com/?cid=2", "placeId": "p2"}},
                ],
                "groundingSupports": [],
            }
        ]
    )
    assert result == GroundingRequests(web_search_requests=None, google_maps_grounding_requests=1)


def test_widget_context_token_alone_counts_one_maps_request():
    result = calculate_grounding_requests([{"googleMapsWidgetContextToken": "widget-token"}])
    assert result == GroundingRequests(web_search_requests=None, google_maps_grounding_requests=1)


def test_combined_web_and_maps_chunks_split_between_both_counters():
    result = calculate_grounding_requests(
        [
            {
                "webSearchQueries": ["q1", "q2"],
                "groundingChunks": [
                    {"web": {"uri": "https://example.com"}},
                    {"maps": {"uri": "https://maps.google.com/?cid=1"}},
                ],
            }
        ]
    )
    assert result == GroundingRequests(web_search_requests=2, google_maps_grounding_requests=1)


def test_url_context_grounding_chunks_without_queries_count_nothing():
    result = calculate_grounding_requests([{"groundingChunks": [{"web": {"uri": "https://example.com"}}]}])
    assert result == GroundingRequests(web_search_requests=None, google_maps_grounding_requests=None)


def test_counters_count_distinct_queries_across_candidates():
    result = calculate_grounding_requests(
        [
            {"webSearchQueries": ["a"]},
            {"groundingChunks": [{"maps": {"uri": "https://maps.google.com/?cid=1"}}]},
            {"webSearchQueries": ["b", "c"], "groundingChunks": [{"maps": {"uri": "https://maps.google.com/?cid=2"}}]},
        ]
    )
    assert result == GroundingRequests(web_search_requests=1, google_maps_grounding_requests=2)


def test_duplicate_queries_across_candidates_collapse_per_bucket():
    result = calculate_grounding_requests(
        [
            {"webSearchQueries": ["shared", "web only"], "groundingChunks": [{"web": {"uri": "https://e.com"}}]},
            {"webSearchQueries": ["shared"], "groundingChunks": [{"web": {"uri": "https://e.com"}}]},
            {"webSearchQueries": ["maps q", "maps q"], "groundingChunks": [{"maps": {"uri": "https://m.com"}}]},
            {"webSearchQueries": ["maps q"], "groundingChunks": [{"maps": {"uri": "https://m.com"}}]},
        ]
    )
    assert result == GroundingRequests(web_search_requests=2, google_maps_grounding_requests=1)


def test_empty_metadata_counts_nothing():
    result = calculate_grounding_requests([])
    assert result == GroundingRequests(web_search_requests=None, google_maps_grounding_requests=None)


def test_has_billable_grounding():
    assert GroundingRequests(web_search_requests=None, google_maps_grounding_requests=1).has_billable_grounding()
    assert GroundingRequests(web_search_requests=1, google_maps_grounding_requests=None).has_billable_grounding()
    assert not GroundingRequests(web_search_requests=None, google_maps_grounding_requests=None).has_billable_grounding()
