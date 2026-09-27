import base64
import struct
from collections.abc import Iterator
from io import BytesIO
from typing import Final

import pytest
from starlette.datastructures import UploadFile

import litellm
import litellm.proxy.proxy_server as proxy_server
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.spend_tracking.budget_reservation import (
    estimate_request_max_cost,
    release_budget_reservation,
    reserve_budget_for_request,
)
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router

MEGAPIXEL: Final = 1024 * 1024
PHOTO_PIXELS: Final = 4032 * 3024


@pytest.fixture
def key_cache() -> Iterator[DualCache]:
    original_counter_cache: Final = proxy_server.spend_counter_cache
    original_key_cache: Final = proxy_server.user_api_key_cache
    original_prisma_client: Final = proxy_server.prisma_client
    cache: Final = DualCache()
    proxy_server.spend_counter_cache = DualCache()
    proxy_server.user_api_key_cache = cache
    proxy_server.prisma_client = None
    try:
        yield cache
    finally:
        proxy_server.spend_counter_cache = original_counter_cache
        proxy_server.user_api_key_cache = original_key_cache
        proxy_server.prisma_client = original_prisma_client


def _png(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
    )


def _jpeg(width: int, height: int, full_metadata_segments: int = 0) -> bytes:
    metadata_segment: Final = b"\xff\xe1" + struct.pack(">H", 0xFFFF) + b"\x00" * 0xFFFD
    return (
        b"\xff\xd8"
        + metadata_segment * full_metadata_segments
        + b"\xff\xc0"
        + struct.pack(">HBHHB", 17, 8, height, width, 3)
        + b"\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    )


def _upload(image: bytes) -> UploadFile:
    return UploadFile(file=BytesIO(image), filename="reference")


def _flux2_pro_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "flux2-pro",
                "litellm_params": {"model": "azure_ai/flux.2-pro", "api_base": "https://foundry.test", "api_key": "k"},
            }
        ]
    )


def _flux2_pro_prices() -> tuple[float, float]:
    catalog_entry: Final = litellm.model_cost["azure_ai/flux.2-pro"]
    return catalog_entry["output_cost_per_image"], catalog_entry["input_cost_per_pixel"] * MEGAPIXEL


async def _reserve(
    key_cache: DualCache,
    key: UserAPIKeyAuth,
    request_body: dict,
    route: str,
    llm_router: Router | None,
    reference_images: tuple[UploadFile, ...] = (),
    fail_closed_budget_enforcement: bool = False,
) -> dict | None:
    return await reserve_budget_for_request(
        request_body=request_body,
        route=route,
        llm_router=llm_router,
        valid_token=key,
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=key_cache,
        proxy_logging_obj=ProxyLogging(user_api_key_cache=key_cache),
        fail_closed_budget_enforcement=fail_closed_budget_enforcement,
        reference_images=reference_images,
    )


async def _cached_key(key_cache: DualCache, token: str, max_budget: float) -> UserAPIKeyAuth:
    key: Final = UserAPIKeyAuth(token=token, spend=0.0, max_budget=max_budget)
    await key_cache.async_set_cache(key=token, value=key)
    return key


@pytest.mark.asyncio
async def test_flux2_edit_with_a_photo_reference_reserves_the_reference_megapixels_and_leaves_the_upload_readable(
    key_cache: DualCache,
):
    first_megapixel, additional_megapixel = _flux2_pro_prices()
    photo: Final = _jpeg(4032, 3024)
    upload: Final = _upload(photo)
    key: Final = await _cached_key(key_cache, "key-flux2-photo", max_budget=1.0)

    reservation: Final = await _reserve(
        key_cache,
        key,
        {"model": "flux2-pro", "prompt": "make it blue", "size": "1024x1024"},
        "/v1/images/edits",
        _flux2_pro_router(),
        reference_images=(upload,),
    )

    assert reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(first_megapixel + 4 * additional_megapixel)
    assert await upload.read() == photo
    await release_budget_reservation(reservation)


@pytest.mark.asyncio
async def test_flux2_edit_is_rejected_when_the_key_only_covers_the_first_megapixel(key_cache: DualCache):
    first_megapixel, _ = _flux2_pro_prices()
    key: Final = await _cached_key(key_cache, "key-flux2-first-megapixel-left", max_budget=first_megapixel)

    with pytest.raises(litellm.BudgetExceededError):
        await _reserve(
            key_cache,
            key,
            {"model": "flux2-pro", "prompt": "make it blue", "size": "1024x1024"},
            "/v1/images/edits",
            _flux2_pro_router(),
            reference_images=(_upload(_jpeg(4032, 3024)),),
            fail_closed_budget_enforcement=True,
        )


@pytest.mark.asyncio
async def test_second_flux2_edit_is_rejected_once_the_first_holds_the_rest_of_the_budget(key_cache: DualCache):
    first_megapixel, additional_megapixel = _flux2_pro_prices()
    key: Final = await _cached_key(
        key_cache, "key-flux2-one-edit-left", max_budget=first_megapixel + 4 * additional_megapixel
    )
    router: Final = _flux2_pro_router()
    request_body: Final = {"model": "flux2-pro", "prompt": "make it blue", "size": "1024x1024"}

    first: Final = await _reserve(
        key_cache, key, request_body, "/v1/images/edits", router, reference_images=(_upload(_jpeg(4032, 3024)),)
    )
    with pytest.raises(litellm.BudgetExceededError):
        await _reserve(
            key_cache, key, request_body, "/v1/images/edits", router, reference_images=(_upload(_jpeg(4032, 3024)),)
        )

    assert first is not None
    await release_budget_reservation(first)


@pytest.mark.parametrize(
    ("request_params", "reference_pixels", "billed_output_megapixels", "billed_reference_megapixels", "images"),
    [
        pytest.param({"size": "1024x1024"}, (MEGAPIXEL, PHOTO_PIXELS), 1, 2, 1, id="several-references-1mp-each"),
        pytest.param({}, (1024 * 1280,), 2, 2, 1, id="no-size-returns-the-reference-size"),
        pytest.param({}, (PHOTO_PIXELS,), 4, 4, 1, id="no-size-output-capped-like-a-lone-reference"),
        pytest.param({"size": "1024x1024"}, (None,), 1, 4, 1, id="unreadable-reference-at-the-lone-maximum"),
        pytest.param({}, (None,), 4, 4, 1, id="unreadable-reference-without-size"),
        pytest.param({"size": "2048x2048"}, (), 4, 0, 1, id="generation-whole-megapixels"),
        pytest.param({"size": "1000x1100"}, (), 2, 0, 1, id="generation-partial-megapixel-rounds-up"),
        pytest.param({}, (), 1, 0, 1, id="generation-default-size"),
        pytest.param({"width": "2048", "height": "1024", "n": "2"}, (), 2, 0, 2, id="form-width-height-and-n"),
        pytest.param({"size": "1024x1024", "num_images": 3}, (), 1, 0, 3, id="num-images"),
    ],
)
def test_flux2_reservation_prices_whole_megapixels_of_output_and_references(
    request_params: dict,
    reference_pixels: tuple[int | None, ...],
    billed_output_megapixels: int,
    billed_reference_megapixels: int,
    images: int,
):
    first_megapixel, additional_megapixel = _flux2_pro_prices()

    reserved: Final = estimate_request_max_cost(
        request_body={"model": "flux2-pro", "prompt": "a cat", **request_params},
        route="/v1/images/edits",
        llm_router=_flux2_pro_router(),
        reference_pixels=reference_pixels,
    )

    image_cost: Final = first_megapixel + additional_megapixel * (billed_output_megapixels - 1)
    assert reserved == pytest.approx(image_cost * images + additional_megapixel * billed_reference_megapixels)


def test_flux2_deployment_own_image_price_is_reserved_flat():
    router: Final = Router(
        model_list=[
            {
                "model_name": "flex-own-price",
                "litellm_params": {
                    "model": "azure_ai/FLUX.2-flex",
                    "api_base": "https://foundry.test",
                    "api_key": "k",
                    "input_cost_per_image": 0.07,
                },
            }
        ]
    )

    reserved: Final = estimate_request_max_cost(
        request_body={"model": "flex-own-price", "prompt": "a cat", "size": "1024x1024"},
        route="/v1/images/edits",
        llm_router=router,
        reference_pixels=(PHOTO_PIXELS,),
    )

    assert reserved == pytest.approx(0.07)


@pytest.mark.asyncio
async def test_flux2_reservation_measures_every_upload_including_a_jpeg_with_large_metadata(key_cache: DualCache):
    first_megapixel, additional_megapixel = _flux2_pro_prices()
    key: Final = await _cached_key(key_cache, "key-flux2-exif", max_budget=1.0)

    reservation: Final = await _reserve(
        key_cache,
        key,
        {"model": "flux2-pro", "prompt": "make it blue"},
        "/v1/images/edits",
        _flux2_pro_router(),
        reference_images=(_upload(_jpeg(1024, 1280, full_metadata_segments=3)),),
    )

    assert reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(first_megapixel + 3 * additional_megapixel)
    await release_budget_reservation(reservation)


@pytest.mark.asyncio
async def test_flux2_passthrough_edit_reserves_its_base64_references(key_cache: DualCache):
    first_megapixel, additional_megapixel = _flux2_pro_prices()
    key: Final = await _cached_key(key_cache, "key-flux2-passthrough", max_budget=1.0)

    reservation: Final = await _reserve(
        key_cache,
        key,
        {
            "prompt": "make it blue",
            "width": 1024,
            "height": 1024,
            "input_image": base64.b64encode(_png(1024, 1024)).decode(),
            "input_image_2": "data:image/png;base64," + base64.b64encode(_png(2048, 2048)).decode(),
        },
        "/azure_ai/flux2-pro/providers/blackforestlabs/v1/flux-2-pro",
        _flux2_pro_router(),
    )

    assert reservation is not None
    assert reservation["reserved_cost"] == pytest.approx(first_megapixel + 2 * additional_megapixel)
    await release_budget_reservation(reservation)


@pytest.mark.parametrize(
    ("model", "route"),
    [
        pytest.param("azure_ai/FLUX.1-Kontext-pro", "/v1/images/edits", id="flux1-kontext-edit"),
        pytest.param("dall-e-3", "/v1/images/generations", id="dall-e-3"),
        pytest.param("openrouter/black-forest-labs/flux.2-pro", "/v1/images/edits", id="flux2-off-azure"),
    ],
)
@pytest.mark.parametrize("routed", [True, False], ids=["router", "no-router"])
def test_references_leave_non_azure_flux2_reservations_unchanged(model: str, route: str, routed: bool):
    router: Final = (
        Router(
            model_list=[
                {
                    "model_name": model,
                    "litellm_params": {
                        "model": model,
                        "api_key": "k",
                        "output_cost_per_image": 0.05,
                        "input_cost_per_pixel": 1e-8,
                    },
                }
            ]
        )
        if routed
        else None
    )
    request_body: Final = {"model": model, "prompt": "a cat", "size": "1024x1024"}

    with_references: Final = estimate_request_max_cost(
        request_body=request_body, route=route, llm_router=router, reference_pixels=(PHOTO_PIXELS, None)
    )

    assert with_references == estimate_request_max_cost(request_body=request_body, route=route, llm_router=router)


def test_flux1_kontext_edit_still_reserves_its_per_image_price():
    reserved: Final = estimate_request_max_cost(
        request_body={"model": "azure_ai/FLUX.1-Kontext-pro", "prompt": "a cat", "n": 2},
        route="/v1/images/edits",
        llm_router=None,
        reference_pixels=(PHOTO_PIXELS,),
    )

    assert reserved == pytest.approx(litellm.model_cost["azure_ai/FLUX.1-Kontext-pro"]["output_cost_per_image"] * 2)


class _ReadCountingBytesIO(BytesIO):
    reads: int = 0

    def read(self, size: int | None = -1) -> bytes:
        self.reads += 1
        return super().read(size)


@pytest.mark.asyncio
async def test_non_flux2_edit_never_reads_its_uploads(key_cache: DualCache):
    upload_file: Final = _ReadCountingBytesIO(_jpeg(4032, 3024))
    key: Final = await _cached_key(key_cache, "key-kontext-upload", max_budget=1.0)
    router: Final = Router(
        model_list=[
            {
                "model_name": "kontext",
                "litellm_params": {"model": "azure_ai/FLUX.1-Kontext-pro", "api_base": "https://foundry.test"},
            }
        ]
    )

    await _reserve(
        key_cache,
        key,
        {"model": "kontext", "prompt": "make it blue"},
        "/v1/images/edits",
        router,
        reference_images=(UploadFile(file=upload_file, filename="photo.jpg"),),
    )

    assert upload_file.reads == 0
