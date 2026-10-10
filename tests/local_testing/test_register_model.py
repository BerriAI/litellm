#### What this tests ####
#    This tests calling batch_completions by running 100 messages together

import ast
from pathlib import Path

import pytest

import litellm


# test_update_model_cost_map_url()


def test_update_model_cost_via_completion():
    try:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey, how's it going?"}],
            input_cost_per_token=0.3,
            output_cost_per_token=0.4,
        )
        print(
            f"litellm.model_cost for gpt-3.5-turbo: {litellm.model_cost['gpt-3.5-turbo']}"
        )
        assert litellm.model_cost["gpt-3.5-turbo"]["input_cost_per_token"] == 0.3
        assert litellm.model_cost["gpt-3.5-turbo"]["output_cost_per_token"] == 0.4
    except Exception as e:
        pytest.fail(f"An error occurred: {e}")


def test_no_test_invocation_at_module_scope():
    tree = ast.parse(Path(__file__).read_text())
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    invoked = [
        node.value.func.id
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id in defined
    ]
    assert not invoked, (
        f"{invoked} run at import time, so pytest collecting this file fires real "
        "provider calls; any failure aborts collection and tears down the whole job"
    )
