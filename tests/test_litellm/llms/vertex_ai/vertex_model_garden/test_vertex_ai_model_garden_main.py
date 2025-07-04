from litellm.llms.vertex_ai.vertex_model_garden.main import create_vertex_url


def test_create_vertex_url():
    assert (
        create_vertex_url(
            vertex_location="us-central1",
            vertex_project="hardy-device-38811",
            stream=False,
            model="deepseek-ai/deepseek-r1-0528-maas",
        )
        == "https://us-central1-aiplatform.googleapis.com/v1/projects/hardy-device-38811/locations/us-central1/endpoints/deepseek-ai/deepseek-r1-0528-maas"
    )
