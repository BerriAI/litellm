"""azure_ai, bedrock, cohere, cohere_chat, dashscope, deepgram, deepseek, groq, mistral, openrouter, perplexity, text-completion-openai, xai cost tracking cases, one CostTrackingTestCase literal per request shape (moved from cost_tracking_cases.json).

PARITY pairs the streamed case with its plain twin where both bill the same row."""

from typing import Final

from integration.cost_calculation.cost_tracking_case import (
    CostTrackingTestCase,
    Deployment,
    EventStreamEvent,
    EventStreamResponse,
    ExactExpected,
    JsonResponse,
    RecountExpected,
    RecountRates,
    SseResponse,
    WavUpload,
)
from integration.cost_calculation.stream_parity.case import StreamParityTestCase, sse_frames


# nova-next
NOVA_NEXT_TRANSCRIPTIONS_PER_SECOND: Final = CostTrackingTestCase(
    name="nova-next-transcriptions-per-second",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="nova-next",
    endpoint="/v1/audio/transcriptions",
    upload=WavUpload(kind="wav", seconds=4.0),
    request={},
    response=JsonResponse(
        content_type="application/json",
        body={
            "results": {"channels": [{"alternatives": [{"transcript": "hello", "confidence": 0.9}]}]},
            "metadata": {"duration": 4.0, "channels": 1},
        },
    ),
    expected=ExactExpected(spend=0.0012, input_cost=0.0012, output_cost=0, prompt_tokens=0, completion_tokens=0),
)


# amazon.nova-canvas-next
AMAZON_NOVA_CANVAS_NEXT_IMAGES_ONE: Final = CostTrackingTestCase(
    name="amazon-nova-canvas-next-images-one",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="amazon.nova-canvas-next",
    endpoint="/v1/images/generations",
    deployment=Deployment(model="amazon.nova-canvas-next"),
    request={"prompt": "a deterministic bedrock image"},
    response=JsonResponse(content_type="application/json", body={"images": ["AA=="]}),
    expected=ExactExpected(
        spend=0.045, input_cost=0.045, output_cost=0, prompt_tokens=0, completion_tokens=0, breakdown_persisted=False
    ),
)


# embed-v5
COHERE_EMBEDDINGS_V5: Final = CostTrackingTestCase(
    name="cohere-embeddings-v5",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="embed-v5",
    endpoint="/v1/embeddings",
    request={"model": "$MODEL", "input": "cohere embedding", "input_type": "search_query"},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "emb-1",
            "embeddings": {"float": [[0.1, 0.2, 0.3]]},
            "meta": {"billed_units": {"input_tokens": 11}},
        },
    ),
    expected=ExactExpected(
        spend=1.144e-05, input_cost=1.144e-05, output_cost=0.0, prompt_tokens=11, completion_tokens=0
    ),
)


# amazon.titan-embed-text-v2:0
BEDROCK_EMBEDDINGS_TITAN_V2: Final = CostTrackingTestCase(
    name="bedrock-embeddings-titan-v2",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="amazon.titan-embed-text-v2:0",
    endpoint="/v1/embeddings",
    request={"model": "$MODEL", "input": "titan embedding"},
    response=JsonResponse(
        content_type="application/json", body={"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 10}
    ),
    expected=ExactExpected(spend=1.05e-05, input_cost=1.05e-05, output_cost=0.0, prompt_tokens=10, completion_tokens=0),
)


# cohere.embed-english-v4
BEDROCK_COHERE_EMBEDDINGS_V4: Final = CostTrackingTestCase(
    name="bedrock-cohere-embeddings-v4",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="cohere.embed-english-v4",
    endpoint="/v1/embeddings",
    request={"model": "$MODEL", "input": "bedrock cohere embedding"},
    response=JsonResponse(
        content_type="application/json",
        body={
            "embeddings": [[0.1, 0.2, 0.3]],
            "id": "emb-bedrock-cohere-1",
            "response_type": "embeddings_floats",
            "texts": ["bedrock cohere embedding"],
        },
    ),
    expected=ExactExpected(spend=5.3e-06, input_cost=5.3e-06, output_cost=0.0, prompt_tokens=5, completion_tokens=0),
)


# rerank-v4
COHERE_RERANK_V4_ONE: Final = CostTrackingTestCase(
    name="cohere-rerank-v4-one",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="rerank-v4",
    endpoint="/v1/rerank",
    request={"model": "$MODEL", "query": "rank this", "documents": ["a", "b"]},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "rr-$REQUEST_ID",
            "results": [{"index": 0, "relevance_score": 0.9}],
            "meta": {"api_version": {"version": "2"}, "billed_units": {"search_units": 1}},
        },
    ),
    expected=ExactExpected(spend=0.0021, input_cost=0.0021, output_cost=0.0, prompt_tokens=0, completion_tokens=0),
)

COHERE_RERANK_V4_THREE: Final = CostTrackingTestCase(
    name="cohere-rerank-v4-three",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="rerank-v4",
    endpoint="/v1/rerank",
    request={
        "model": "$MODEL",
        "query": "rank this",
        "documents": ["a long document", "another long document", "third long document"],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "rr-three-$REQUEST_ID",
            "results": [{"index": 0, "relevance_score": 0.9}],
            "meta": {"api_version": {"version": "2"}, "billed_units": {"search_units": 3}},
        },
    ),
    expected=ExactExpected(spend=0.0063, input_cost=0.0063, output_cost=0.0, prompt_tokens=0, completion_tokens=0),
)

COHERE_RERANK_V4_TOTAL_TOKENS_FALLBACK: Final = CostTrackingTestCase(
    name="cohere-rerank-v4-total-tokens-fallback",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="rerank-v4",
    endpoint="/v1/rerank",
    request={"model": "$MODEL", "query": "rank this", "documents": ["fallback a", "fallback b"]},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "rr-fallback-$REQUEST_ID",
            "results": [{"index": 0, "relevance_score": 0.8}],
            "meta": {"billed_units": {"total_tokens": 99}},
        },
    ),
    expected=ExactExpected(spend=0.0, input_cost=0.0, output_cost=0.0, prompt_tokens=0, completion_tokens=0),
)


# cohere.rerank-v4:0
BEDROCK_COHERE_RERANK_V4: Final = CostTrackingTestCase(
    name="bedrock-cohere-rerank-v4",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="cohere.rerank-v4:0",
    endpoint="/v1/rerank",
    request={"model": "$MODEL", "query": "rank this", "documents": ["a", "b"]},
    response=JsonResponse(
        content_type="application/json",
        body={"results": [{"index": 0, "relevanceScore": 0.9}], "response_id": "rr-3", "token_count": 1},
    ),
    expected=ExactExpected(spend=0.0022, input_cost=0.0022, output_cost=0.0, prompt_tokens=0, completion_tokens=0),
)


# gpt-3.5-turbo-instruct-next
TEXT_COMPLETIONS_OPENAI_BASIC: Final = CostTrackingTestCase(
    name="text-completions-openai-basic",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-3.5-turbo-instruct-next",
    endpoint="/v1/completions",
    request={"model": "$MODEL", "prompt": "complete this"},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "cmpl-basic-$REQUEST_ID",
            "object": "text_completion",
            "choices": [{"text": "done", "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
        },
    ),
    expected=ExactExpected(
        spend=1.882e-05, input_cost=1.026e-05, output_cost=8.56e-06, prompt_tokens=9, completion_tokens=4
    ),
)

TEXT_COMPLETIONS_OPENAI_STREAM_USAGE: Final = CostTrackingTestCase(
    name="text-completions-openai-stream-usage",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-3.5-turbo-instruct-next",
    endpoint="/v1/completions",
    request={"model": "$MODEL", "prompt": "complete this", "stream": True, "stream_options": {"include_usage": True}},
    response=SseResponse(
        content_type="text/event-stream",
        frames=sse_frames(
            {
                "id": "cmpl-$REQUEST_ID",
                "object": "text_completion",
                "created": 1789789000,
                "model": "gpt-3.5-turbo-instruct-next",
                "choices": [{"text": "done", "index": 0, "finish_reason": None}],
                "usage": None,
            },
            {
                "id": "cmpl-$REQUEST_ID",
                "object": "text_completion",
                "created": 1789789000,
                "model": "gpt-3.5-turbo-instruct-next",
                "choices": [],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
            },
            done=True,
        ),
    ),
    expected=ExactExpected(
        spend=1.882e-05, input_cost=1.026e-05, output_cost=8.56e-06, prompt_tokens=9, completion_tokens=4
    ),
)

TEXT_COMPLETIONS_OPENAI_N_BEST: Final = CostTrackingTestCase(
    name="text-completions-openai-n-best",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="gpt-3.5-turbo-instruct-next",
    endpoint="/v1/completions",
    request={"model": "$MODEL", "prompt": "complete this twice", "n": 2},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "cmpl-n-best-$REQUEST_ID",
            "object": "text_completion",
            "choices": [
                {"text": "done", "index": 0, "finish_reason": "stop"},
                {"text": "also done", "index": 1, "finish_reason": "stop"},
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 8, "total_tokens": 17},
        },
    ),
    expected=ExactExpected(
        spend=2.738e-05, input_cost=1.026e-05, output_cost=1.712e-05, prompt_tokens=9, completion_tokens=8
    ),
)


# dashscope/qwen4-max
DASHSCOPE_QWEN4_MAX_TIERED_INPUT: Final = CostTrackingTestCase(
    name="dashscope-qwen4-max-tiered_input",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="dashscope/qwen4-max",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "tiered input"}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "$REQUEST_ID",
            "object": "chat.completion",
            "model": "qwen4-max",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.00507, input_cost=0.002392, output_cost=0.002678, prompt_tokens=1840, completion_tokens=412
    ),
)

DASHSCOPE_QWEN4_MAX_TIERED_BOUNDARY_STAYS_LOWER_TIER: Final = CostTrackingTestCase(
    name="dashscope-qwen4-max-tiered_boundary_stays_lower_tier",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="dashscope/qwen4-max",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "tier boundary"}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "$REQUEST_ID",
            "object": "chat.completion",
            "model": "qwen4-max",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 32000, "completion_tokens": 412, "total_tokens": 32412},
        },
    ),
    expected=ExactExpected(
        spend=0.044278, input_cost=0.0416, output_cost=0.002678, prompt_tokens=32000, completion_tokens=412
    ),
)

DASHSCOPE_QWEN4_MAX_TIERED_SECOND_TIER: Final = CostTrackingTestCase(
    name="dashscope-qwen4-max-tiered_second_tier",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="dashscope/qwen4-max",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "tier two"}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "$REQUEST_ID",
            "object": "chat.completion",
            "model": "qwen4-max",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40000, "completion_tokens": 412, "total_tokens": 40412},
        },
    ),
    expected=ExactExpected(
        spend=0.109356, input_cost=0.104, output_cost=0.005356, prompt_tokens=40000, completion_tokens=412
    ),
)

DASHSCOPE_QWEN4_MAX_TIERED_ABOVE_TOP_RANGE: Final = CostTrackingTestCase(
    name="dashscope-qwen4-max-tiered_above_top_range",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="dashscope/qwen4-max",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "top tier"}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "$REQUEST_ID",
            "object": "chat.completion",
            "model": "qwen4-max",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 300000, "completion_tokens": 412, "total_tokens": 300412},
        },
    ),
    expected=ExactExpected(
        spend=0.936386, input_cost=0.93, output_cost=0.006386, prompt_tokens=300000, completion_tokens=412
    ),
)


# openrouter/anthropic/claude-sonnet-5
OPENROUTER_ANTHROPIC_CLAUDE_SONNET_5_PROVIDER_REPORTED_COST: Final = CostTrackingTestCase(
    name="openrouter-anthropic-claude-sonnet-5-provider_reported_cost",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="openrouter/anthropic/claude-sonnet-5",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "reported cost"}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "model": "anthropic/claude-sonnet-5",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252, "cost": 0.0421},
        },
    ),
    expected=ExactExpected(
        spend=0.0421,
        input_cost=0.0,
        output_cost=0.0421,
        prompt_tokens=1840,
        completion_tokens=412,
        breakdown_persisted=False,
    ),
)

OPENROUTER_ANTHROPIC_CLAUDE_SONNET_5_TOKEN_PRICED: Final = CostTrackingTestCase(
    name="openrouter-anthropic-claude-sonnet-5-token_priced",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="openrouter/anthropic/claude-sonnet-5",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "token pricing"}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "model": "anthropic/claude-sonnet-5",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.01248, input_cost=0.005888, output_cost=0.006592, prompt_tokens=1840, completion_tokens=412
    ),
)


# perplexity/sonar-next
PERPLEXITY_SONAR_NEXT_NO_SEARCH: Final = CostTrackingTestCase(
    name="perplexity-sonar-next-no_search",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="perplexity/sonar-next",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "no search"}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "model": "sonar-next",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.0025118, input_cost=0.0020792, output_cost=0.0004326, prompt_tokens=1840, completion_tokens=412
    ),
)


# deepseek/deepseek-v4-chat
DEEPSEEK_DEEPSEEK_V4_CHAT_PROMPT_CACHE_HIT: Final = CostTrackingTestCase(
    name="deepseek-deepseek-v4-chat-prompt_cache_hit",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="deepseek/deepseek-v4-chat",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "cache hit"}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "model": "deepseek-v4-chat",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1840,
                "completion_tokens": 412,
                "total_tokens": 2252,
                "prompt_cache_hit_tokens": 1200,
                "prompt_cache_miss_tokens": 640,
                "prompt_tokens_details": {"cached_tokens": 1200},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.00039756,
        input_cost=0.0002204,
        output_cost=0.00017716,
        cache_read_cost=3.48e-05,
        prompt_tokens=1840,
        completion_tokens=412,
    ),
)

DEEPSEEK_DEEPSEEK_V4_CHAT_NO_CACHE_FIELDS_BILLS_ZERO_CACHE: Final = CostTrackingTestCase(
    name="deepseek-deepseek-v4-chat-no_cache_fields_bills_zero_cache",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="deepseek/deepseek-v4-chat",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "no cache"}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "model": "deepseek-v4-chat",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.00071076,
        input_cost=0.0005336,
        output_cost=0.00017716,
        cache_read_cost=0.0,
        prompt_tokens=1840,
        completion_tokens=412,
    ),
)


# xai/grok-5
XAI_GROK_5_REASONING_FOLDED_INTO_COMPLETION: Final = CostTrackingTestCase(
    name="xai-grok-5-reasoning_folded_into_completion",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="xai/grok-5",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "reasoning"}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "model": "grok-5",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1840,
                "completion_tokens": 412,
                "total_tokens": 2552,
                "completion_tokens_details": {"reasoning_tokens": 300},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.0044064, input_cost=0.002484, output_cost=0.0019224, prompt_tokens=1840, completion_tokens=712
    ),
)

XAI_GROK_5_LIVE_SEARCH: Final = CostTrackingTestCase(
    name="xai-grok-5-live_search",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="xai/grok-5",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "live search"}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "model": "grok-5",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1840,
                "completion_tokens": 412,
                "total_tokens": 2252,
                "server_side_tool_usage_details": {"web_search_calls": 2},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.0135964,
        input_cost=0.002484,
        output_cost=0.0011124,
        tool_usage_cost=0.01,
        prompt_tokens=1840,
        completion_tokens=412,
    ),
)

XAI_GROK_5_PROVIDER_REPORTED_COST: Final = CostTrackingTestCase(
    name="xai-grok-5-provider_reported_cost",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="xai/grok-5",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "reported xai cost"}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "model": "grok-5",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252, "cost": 0.0421},
        },
    ),
    expected=ExactExpected(spend=0.0421, input_cost=0.0, output_cost=0.0421, prompt_tokens=1840, completion_tokens=412),
)


# bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0
BEDROCK_INVOKE_HAIKU_JSON: Final = CostTrackingTestCase(
    name="bedrock-invoke-haiku-json",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
    endpoint="/v1/messages",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "bedrock-invoke-haiku-json"}],
        "stream": False,
        "max_tokens": 412,
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "msg_$REQUEST_ID",
            "type": "message",
            "role": "assistant",
            "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
            "content": [{"type": "text", "text": "scripted response"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1840, "output_tokens": 412},
        },
    ),
    expected=ExactExpected(
        spend=0.00425372, input_cost=0.0021896, output_cost=0.00206412, prompt_tokens=1840, completion_tokens=412
    ),
)

BEDROCK_INVOKE_HAIKU_STREAM: Final = CostTrackingTestCase(
    name="bedrock-invoke-haiku-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
    endpoint="/v1/messages",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "bedrock-invoke-haiku-stream"}],
        "stream": True,
        "max_tokens": 412,
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(
                event_type="message_start",
                payload={
                    "type": "message_start",
                    "message": {
                        "id": "msg_$REQUEST_ID",
                        "type": "message",
                        "role": "assistant",
                        "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1840, "output_tokens": 0},
                    },
                },
            ),
            EventStreamEvent(
                event_type="content_block_delta",
                payload={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "scripted response"},
                },
            ),
            EventStreamEvent(
                event_type="message_delta",
                payload={
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 412},
                },
            ),
            EventStreamEvent(event_type="message_stop", payload={"type": "message_stop"}),
        ),
        framing="invoke",
    ),
    expected=ExactExpected(
        spend=0.00425372, input_cost=0.0021896, output_cost=0.00206412, prompt_tokens=1840, completion_tokens=412
    ),
)


# azure_ai/gpt-5.4-mini-2026-03-17
AZURE_AI_GPT_5_4_MINI_LATEST: Final = CostTrackingTestCase(
    name="azure-ai-gpt-5.4-mini-latest",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="azure_ai/gpt-5.4-mini-2026-03-17",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "azure-ai-gpt-5.4-mini-latest"}],
        "stream": False,
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "model": "$MODEL",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "scripted response"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            "service_tier": "default",
        },
    ),
    expected=ExactExpected(
        spend=0.003234, input_cost=0.00138, output_cost=0.001854, prompt_tokens=1840, completion_tokens=412
    ),
)

AZURE_AI_GPT_5_4_MINI_LATEST_STREAM: Final = CostTrackingTestCase(
    name="azure-ai-gpt-5.4-mini-latest-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="azure_ai/gpt-5.4-mini-2026-03-17",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "azure-ai-gpt-5.4-mini-latest-stream"}],
        "stream": True,
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=(
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1,"model":"$MODEL","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1,"model":"$MODEL","choices":[],"usage":{"prompt_tokens":1840,"completion_tokens":412,"total_tokens":2252}}\n\n',
            "data: [DONE]\n\n",
        ),
    ),
    expected=ExactExpected(
        spend=0.003234, input_cost=0.00138, output_cost=0.001854, prompt_tokens=1840, completion_tokens=412
    ),
)


# groq/qwen/qwen3.8-27b
GROQ_QWEN_3_8_JSON: Final = CostTrackingTestCase(
    name="groq-qwen-3.8-json",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="groq/qwen/qwen3.8-27b",
    request={"model": "$MODEL", "messages": [{"role": "user", "content": "groq-qwen-3.8-json"}], "stream": False},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "model": "$MODEL",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "scripted response"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            "service_tier": "default",
            "x_groq": {"usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252}},
        },
    ),
    expected=ExactExpected(
        spend=0.00312, input_cost=0.001472, output_cost=0.001648, prompt_tokens=1840, completion_tokens=412
    ),
)

GROQ_QWEN_3_8_STREAM_X_GROQ_RECOUNT: Final = CostTrackingTestCase(
    name="groq-qwen-3.8-stream_x_groq_recount",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="groq/qwen/qwen3.8-27b",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "groq-qwen-3.8-stream_x_groq_recount"}],
        "stream": True,
    },
    response=SseResponse(
        content_type="text/event-stream",
        frames=(
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1,"model":"$MODEL","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}\n\n',
            'data: {"id":"chatcmpl-$REQUEST_ID","object":"chat.completion.chunk","created":1,"model":"$MODEL","choices":[],"x_groq":{"usage":{"prompt_tokens":1840,"completion_tokens":412,"total_tokens":2252}}}\n\n',
            "data: [DONE]\n\n",
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=8e-07, output_cost_per_token=4e-06)),
)


# cohere_chat/v2/command-a-03-2025
COHERE_COMMAND_A_V2_TOKENS: Final = CostTrackingTestCase(
    name="cohere-command-a-v2-tokens",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="cohere_chat/v2/command-a-03-2025",
    deployment=Deployment(model="cohere_chat/v2/command-a-03-2025"),
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "cohere-command-a-v2-tokens"}],
        "stream": False,
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "$REQUEST_ID",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "scripted response"}]},
            "finish_reason": "COMPLETE",
            "usage": {
                "tokens": {"input_tokens": 1840, "output_tokens": 412, "total_tokens": 2252},
                "billed_units": {"input_tokens": 1800, "output_tokens": 400},
            },
        },
    ),
    expected=ExactExpected(
        spend=0.00874252, input_cost=0.0046184, output_cost=0.00412412, prompt_tokens=1840, completion_tokens=412
    ),
)


# mistral/mistral-medium-2604
MISTRAL_MEDIUM_2604_JSON: Final = CostTrackingTestCase(
    name="mistral-medium-2604-json",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="mistral/mistral-medium-2604",
    request={"model": "$MODEL", "messages": [{"role": "user", "content": "mistral-medium-2604-json"}], "stream": False},
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "model": "$MODEL",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "scripted response"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
            "service_tier": "default",
        },
    ),
    expected=ExactExpected(
        spend=0.00587252, input_cost=0.0027784, output_cost=0.00309412, prompt_tokens=1840, completion_tokens=412
    ),
)


# perplexity/pplx-decider-v1-27b
PERPLEXITY_PPLX_DECIDER_V1_27B_DECISIONS: Final = CostTrackingTestCase(
    name="perplexity/pplx-decider-v1-27b-decisions",
    covers="quota_management.spend_tracking.decisions_costs",
    model="perplexity/pplx-decider-v1-27b",
    endpoint="/v1/systemone",
    request={
        "model": "$MODEL",
        "state": {"source": "cost-tracking"},
        "questions": {"is_defect": {"type": "noul", "instructions": "Is this a defect?"}},
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "model": "pplx-decider-v1-27b",
            "answers": {"is_defect": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 367, "output_tokens": 3},
        },
    ),
    expected=ExactExpected(
        spend=1.468e-05, input_cost=1.468e-05, output_cost=0.0, prompt_tokens=367, completion_tokens=3
    ),
)


# xai/grok-4.7
XAI_GROK_4_7_INPUT_TEXT: Final = CostTrackingTestCase(
    name="xai-grok-4.7-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="xai/grok-4.7",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "Say hello."}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "id": "chatcmpl-$REQUEST_ID",
            "object": "chat.completion",
            "model": "grok-4.7",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1840, "completion_tokens": 412, "total_tokens": 2252},
        },
    ),
    expected=ExactExpected(
        spend=0.006152, input_cost=0.00368, output_cost=0.002472, prompt_tokens=1840, completion_tokens=412
    ),
)


CASES: Final[tuple[CostTrackingTestCase, ...]] = (
    NOVA_NEXT_TRANSCRIPTIONS_PER_SECOND,
    AMAZON_NOVA_CANVAS_NEXT_IMAGES_ONE,
    COHERE_EMBEDDINGS_V5,
    BEDROCK_EMBEDDINGS_TITAN_V2,
    BEDROCK_COHERE_EMBEDDINGS_V4,
    COHERE_RERANK_V4_ONE,
    COHERE_RERANK_V4_THREE,
    COHERE_RERANK_V4_TOTAL_TOKENS_FALLBACK,
    BEDROCK_COHERE_RERANK_V4,
    TEXT_COMPLETIONS_OPENAI_BASIC,
    TEXT_COMPLETIONS_OPENAI_STREAM_USAGE,
    TEXT_COMPLETIONS_OPENAI_N_BEST,
    DASHSCOPE_QWEN4_MAX_TIERED_INPUT,
    DASHSCOPE_QWEN4_MAX_TIERED_BOUNDARY_STAYS_LOWER_TIER,
    DASHSCOPE_QWEN4_MAX_TIERED_SECOND_TIER,
    DASHSCOPE_QWEN4_MAX_TIERED_ABOVE_TOP_RANGE,
    OPENROUTER_ANTHROPIC_CLAUDE_SONNET_5_PROVIDER_REPORTED_COST,
    OPENROUTER_ANTHROPIC_CLAUDE_SONNET_5_TOKEN_PRICED,
    PERPLEXITY_SONAR_NEXT_NO_SEARCH,
    DEEPSEEK_DEEPSEEK_V4_CHAT_PROMPT_CACHE_HIT,
    DEEPSEEK_DEEPSEEK_V4_CHAT_NO_CACHE_FIELDS_BILLS_ZERO_CACHE,
    XAI_GROK_5_REASONING_FOLDED_INTO_COMPLETION,
    XAI_GROK_5_LIVE_SEARCH,
    XAI_GROK_5_PROVIDER_REPORTED_COST,
    BEDROCK_INVOKE_HAIKU_JSON,
    BEDROCK_INVOKE_HAIKU_STREAM,
    AZURE_AI_GPT_5_4_MINI_LATEST,
    AZURE_AI_GPT_5_4_MINI_LATEST_STREAM,
    GROQ_QWEN_3_8_JSON,
    GROQ_QWEN_3_8_STREAM_X_GROQ_RECOUNT,
    COHERE_COMMAND_A_V2_TOKENS,
    MISTRAL_MEDIUM_2604_JSON,
    PERPLEXITY_PPLX_DECIDER_V1_27B_DECISIONS,
    XAI_GROK_4_7_INPUT_TEXT,
)

PARITY: Final[tuple[StreamParityTestCase, ...]] = (
    StreamParityTestCase(plain=TEXT_COMPLETIONS_OPENAI_BASIC, streamed=TEXT_COMPLETIONS_OPENAI_STREAM_USAGE),
)
