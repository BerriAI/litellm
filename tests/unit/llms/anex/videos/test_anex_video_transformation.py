"""
Tests for anex.sh video generation, which serves the MiniMax V2 video task API under its own host, key and prices.
"""

import json
from collections.abc import Callable
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.anex.videos.transformation import ANEX_VIDEO_DEFAULT_API_BASE, get_anex_video_config
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai.cost_calculation import video_generation_cost
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider
from litellm.videos.main import avideo_content, avideo_generation, avideo_status

EU_API_BASE: Final = "https://tasks-eu.anex.sh"
ANEX_VIDEO_MODELS: Final = ("MiniMax-H3", "MiniMax-H3-Ref2VA", "MiniMax-H3-Ref2VA-SMD-4step", "FastH3-8-Step-V2")
IMAGE_BILLED_MODELS: Final = ("MiniMax-H3", "MiniMax-H3-Ref2VA", "MiniMax-H3-Ref2VA-SMD-4step")
TIERS: Final = ("768p", "2k", "4k")


async def _call(
    respond: Callable[[httpx.Request], httpx.Response], call: Callable[..., object], **kwargs: object
) -> tuple[object, list[httpx.Request]]:
    sent: Final[list[httpx.Request]] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return respond(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(record)) as http_client:
        handler: Final = AsyncHTTPHandler()
        await handler.close()
        handler.client = http_client
        result: Final = await call(client=handler, **kwargs)
    return result, sent


def _task_created(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"task_id": "318407566120873452"})


def _priced_tiers(model: str) -> tuple[str, ...]:
    info: Final = litellm.get_model_info(model=model, custom_llm_provider="anex")
    return tuple(tier for tier in TIERS if info.get(f"output_cost_per_second_{tier}") is not None)


class TestAnexVideoRouting:
    @pytest.mark.asyncio
    async def test_create_goes_to_the_anex_task_api_with_the_anex_key(self, monkeypatch):
        monkeypatch.setenv("ANEX_API_KEY", "anex-env-key")
        monkeypatch.delenv("ANEX_API_BASE", raising=False)

        video, sent = await _call(
            _task_created, avideo_generation, model="anex/MiniMax-H3", prompt="A red fox trots across fresh snow"
        )

        (request,) = sent
        assert str(request.url) == f"{ANEX_VIDEO_DEFAULT_API_BASE}/v2/video_generation"
        assert request.headers["Authorization"] == "Bearer anex-env-key"
        body: Final = json.loads(request.content)
        assert body["model"] == "MiniMax-H3"
        assert body["content"] == [{"type": "text", "text": "A red fox trots across fresh snow"}]
        decoded: Final = decode_video_id_with_provider(video.id)
        assert decoded["custom_llm_provider"] == "anex"
        assert decoded["video_id"] == "318407566120873452"
        assert decoded["model_id"] == "MiniMax-H3"

    @pytest.mark.asyncio
    async def test_the_env_api_base_moves_every_request_to_the_eu_host(self, monkeypatch):
        monkeypatch.setenv("ANEX_API_BASE", EU_API_BASE)

        _, sent = await _call(_task_created, avideo_generation, model="anex/MiniMax-H3", prompt="p", api_key="k")

        assert str(sent[0].url) == f"{EU_API_BASE}/v2/video_generation"

    @pytest.mark.asyncio
    async def test_an_explicit_api_base_wins_over_the_env_api_base(self, monkeypatch):
        monkeypatch.setenv("ANEX_API_BASE", "https://tasks-us.anex.sh")

        _, sent = await _call(
            _task_created, avideo_generation, model="anex/MiniMax-H3", prompt="p", api_key="k", api_base=EU_API_BASE
        )

        assert str(sent[0].url) == f"{EU_API_BASE}/v2/video_generation"

    def test_a_minimax_key_is_never_sent_to_anex(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "minimax-key")
        monkeypatch.delenv("ANEX_API_KEY", raising=False)
        monkeypatch.setattr("litellm.api_key", None)

        with pytest.raises(ValueError, match="ANEX_API_KEY"):
            get_anex_video_config().validate_environment(
                headers={}, model="MiniMax-H3", litellm_params=GenericLiteLLMParams()
            )

    @pytest.mark.asyncio
    async def test_seed_reaches_the_request_body(self):
        _, sent = await _call(
            _task_created, avideo_generation, model="anex/FastH3-8-Step-V2", prompt="p", seed=42, api_key="k"
        )

        assert json.loads(sent[0].content)["seed"] == 42

    @pytest.mark.asyncio
    async def test_a_param_anex_does_not_take_is_rejected_as_an_anex_error_before_the_billed_call(self):
        with pytest.raises(litellm.UnsupportedParamsError, match=r"anex\.sh") as raised:
            await _call(
                _task_created, avideo_generation, model="anex/MiniMax-H3", prompt="p", negative_prompt="x", api_key="k"
            )

        assert raised.value.llm_provider == "anex"


class TestAnexVideoPollAndDownload:
    @pytest.mark.asyncio
    async def test_status_polls_the_task_and_keeps_the_anex_id(self):
        encoded: Final = encode_video_id_with_provider("318407566120873452", "anex", "MiniMax-H3")

        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "task": {
                        "id": "318407566120873452",
                        "model": "MiniMax-H3",
                        "status": "succeeded",
                        "created_at": 1761750000,
                        "updated_at": 1761750210,
                        "content": {"url": "https://media.example.com/clip.mp4"},
                        "resolution": "768P",
                        "duration": 6,
                        "usage": {"total_seconds": 6, "input_seconds": 0, "output_seconds": 6, "input_image_count": 0},
                    }
                },
            )

        video, sent = await _call(respond, avideo_status, video_id=encoded, api_key="k")

        assert str(sent[0].url) == f"{ANEX_VIDEO_DEFAULT_API_BASE}/v2/query/video_generation/318407566120873452"
        assert video.id == encoded
        assert video.status == "completed"
        assert video.seconds == "6"

    @pytest.mark.asyncio
    async def test_content_downloads_the_clip_the_task_links_to(self, monkeypatch):
        monkeypatch.setattr(litellm, "user_url_validation", False)
        encoded: Final = encode_video_id_with_provider("318407566120873452", "anex", "MiniMax-H3")
        clip_url: Final = "https://media.example.com/clip.mp4"

        def respond(request: httpx.Request) -> httpx.Response:
            if str(request.url) == clip_url:
                return httpx.Response(200, content=b"mp4-bytes")
            return httpx.Response(
                200, json={"task": {"id": "318407566120873452", "status": "succeeded", "content": {"url": clip_url}}}
            )

        content, sent = await _call(respond, avideo_content, video_id=encoded, api_key="k")

        assert content == b"mp4-bytes"
        assert [str(request.url) for request in sent] == [
            f"{ANEX_VIDEO_DEFAULT_API_BASE}/v2/query/video_generation/318407566120873452",
            clip_url,
        ]


@pytest.mark.usefixtures("local_model_cost_map")
class TestAnexVideoPricing:
    @pytest.mark.parametrize("model", ANEX_VIDEO_MODELS)
    def test_every_priced_tier_bills_its_own_rate(self, model):
        info: Final = litellm.get_model_info(model=model, custom_llm_provider="anex")
        tiers: Final = _priced_tiers(model)

        assert "768p" in tiers
        for tier in tiers:
            cost = video_generation_cost(
                model=model, duration_seconds=6.0, custom_llm_provider="anex", video_resolution=tier
            )
            assert cost == pytest.approx(info[f"output_cost_per_second_{tier}"] * 6.0)

    @pytest.mark.parametrize("model", ANEX_VIDEO_MODELS)
    def test_a_higher_resolution_never_bills_less(self, model):
        info: Final = litellm.get_model_info(model=model, custom_llm_provider="anex")
        rates: Final = [info[f"output_cost_per_second_{tier}"] for tier in _priced_tiers(model)]

        assert all(rate > 0 for rate in rates)
        assert rates == sorted(rates)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ANEX_VIDEO_MODELS)
    async def test_a_resolution_without_a_price_is_rejected_before_the_billed_call(self, model):
        unpriced: Final = next(tier for tier in ("480p", *TIERS) if tier not in _priced_tiers(model))
        sent: Final[list[httpx.Request]] = []

        def respond(request: httpx.Request) -> httpx.Response:
            sent.append(request)
            return _task_created(request)

        with pytest.raises(litellm.BadRequestError, match=unpriced.upper()) as raised:
            await _call(
                respond, avideo_generation, model=f"anex/{model}", prompt="p", resolution=unpriced.upper(), api_key="k"
            )

        assert raised.value.llm_provider == "anex"
        assert sent == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ANEX_VIDEO_MODELS)
    async def test_every_priced_resolution_is_sent_as_asked(self, model):
        for tier in _priced_tiers(model):
            _, sent = await _call(
                _task_created,
                avideo_generation,
                model=f"anex/{model}",
                prompt="p",
                resolution=tier.upper(),
                api_key="k",
            )
            assert json.loads(sent[0].content)["resolution"] == tier.upper()

    @staticmethod
    def _create_cost(model: str, image_count: int, resolution: str = "768P") -> float:
        content: Final = [
            {"type": "text", "text": "p"},
            *(
                {"type": "image_url", "image_url": {"url": f"https://x/{i}.png"}, "role": "reference_image"}
                for i in range(image_count)
            ),
        ]
        video: Final = get_anex_video_config().transform_video_create_response(
            model=model,
            raw_response=httpx.Response(200, json={"task_id": "t1"}),
            logging_obj=None,
            custom_llm_provider="anex",
            request_data={"model": model, "content": content, "duration": 6, "resolution": resolution},
        )
        return litellm.completion_cost(
            completion_response=video, model=f"anex/{model}", call_type="create_video", custom_llm_provider="anex"
        )

    @pytest.mark.parametrize("model", IMAGE_BILLED_MODELS)
    def test_images_beyond_the_free_allowance_are_billed_per_image(self, model):
        info: Final = litellm.get_model_info(model=model, custom_llm_provider="anex")
        free_images: Final = info["provider_specific_entry"]["anex_free_input_images"]
        output_only: Final = self._create_cost(model, image_count=0)

        assert output_only == pytest.approx(info["output_cost_per_second_768p"] * 6)
        assert self._create_cost(model, image_count=free_images) == pytest.approx(output_only)
        assert self._create_cost(model, image_count=free_images + 3) == pytest.approx(
            output_only + 3 * info["input_cost_per_image"]
        )

    @pytest.mark.parametrize("model", IMAGE_BILLED_MODELS)
    def test_a_create_at_a_higher_resolution_is_billed_at_that_tier(self, model):
        info: Final = litellm.get_model_info(model=model, custom_llm_provider="anex")

        assert self._create_cost(model, image_count=0, resolution="4K") == pytest.approx(
            info["output_cost_per_second_4k"] * 6
        )
