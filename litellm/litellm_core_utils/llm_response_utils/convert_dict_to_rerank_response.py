from typing import Dict, Optional

from litellm.types.rerank import RerankResponse


def convert_dict_to_rerank_response(
    model_response_object: Optional[RerankResponse],
    response_object: Optional[Dict],
) -> RerankResponse:
    if response_object is None:
        raise Exception("Error in response object format")

    if model_response_object is None:
        model_response_object = RerankResponse(**response_object)
        return model_response_object

    if "id" in response_object:
        model_response_object.id = response_object["id"]

    if "meta" in response_object:
        model_response_object.meta = response_object["meta"]

    if "results" in response_object:
        model_response_object.results = response_object["results"]

    return model_response_object
