import os
import litellm

def test_model_cost_map_not_called_on_import_then_called_on_model_cost():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    assert litellm._model_cost is None
    model_cost = litellm.model_cost()
    assert model_cost is not None
    assert litellm._model_cost == model_cost