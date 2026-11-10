from litellm.llms.minimax.image_edit.transformation import MinimaxImageEditConfig


def test_minimax_image_edit_sends_json_image_url():
    config = MinimaxImageEditConfig()
    assert config.use_multipart_form_data() is False
    assert (
        config.get_complete_url(
            model="MiniMax-H3-Image", api_base=None, litellm_params={}
        )
        == "https://api.minimax.io/v1/images/edits"
    )
    body, files = config.transform_image_edit_request(
        model="MiniMax-H3-Image",
        prompt="make it night",
        image="https://example.com/ref.png",
        image_edit_optional_request_params={"aspect_ratio": "1:1", "area": "1K"},
        litellm_params=None,
        headers={},
    )
    assert files == []
    assert body == {
        "model": "MiniMax-H3-Image",
        "images": [{"image_url": "https://example.com/ref.png"}],
        "aspect_ratio": "1:1",
        "area": "1K",
        "prompt": "make it night",
    }
