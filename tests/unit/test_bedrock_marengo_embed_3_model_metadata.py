from litellm.constants import bedrock_embedding_models

BASE_MODEL = "twelvelabs.marengo-embed-3-0-v1:0"

TEXT_REQUEST_COST = 7e-05
IMAGE_REQUEST_COST = 0.0001
VIDEO_COST_PER_SECOND = 0.0007
AUDIO_COST_PER_SECOND = 0.00014


def test_marengo_embed_3_is_a_known_bedrock_embedding_model():
    assert BASE_MODEL in bedrock_embedding_models
