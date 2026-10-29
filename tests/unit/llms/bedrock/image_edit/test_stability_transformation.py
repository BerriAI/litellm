from litellm.llms.bedrock.image_edit.stability_transformation import BedrockStabilityImageEditConfig


def test_transform_image_edit_request_returns_the_json_body_and_no_files():
    request = BedrockStabilityImageEditConfig().transform_image_edit_request(
        model="stability.stable-image-inpaint-v1:0",
        prompt="add a red hat",
        image=b"\x89PNG-bytes",
        image_edit_optional_request_params={},
        litellm_params={},
        headers={},
    )

    assert request == ({"output_format": "png", "prompt": "add a red hat", "image": "iVBORy1ieXRlcw=="}, {})
