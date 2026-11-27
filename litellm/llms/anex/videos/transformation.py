"""anex.sh video generation. Its task API at tasks.anex.sh follows the MiniMax V2 video task API."""

from typing import Final

from litellm.llms.minimax.videos.transformation import MinimaxVideoConfig, MinimaxVideoProvider

ANEX_VIDEO_DEFAULT_API_BASE: Final = "https://tasks.anex.sh"

ANEX_VIDEO_PROVIDER: Final = MinimaxVideoProvider(
    slug="anex",
    display_name="anex.sh",
    api_key_env="ANEX_API_KEY",
    api_base_env="ANEX_API_BASE",
    default_api_base=ANEX_VIDEO_DEFAULT_API_BASE,
    extra_create_body_keys=("seed",),
)


def get_anex_video_config() -> MinimaxVideoConfig:
    return MinimaxVideoConfig(ANEX_VIDEO_PROVIDER)
