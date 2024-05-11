# What is this?
## Test to make sure function call response always works with json.loads() -> no extra parsing required. Relevant issue - https://github.com/BerriAI/litellm/issues/2654
import sys, os
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, io

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
import json
import warnings

from litellm import completion
from typing import List


# Just a stub to keep the sample code simple
class Trade:
    def __init__(self, order: dict):
        self.order = order

    @staticmethod
    def buy(order: dict):
        return Trade(order)

    @staticmethod
    def sell(order: dict):
        return Trade(order)


def trade(model_name: str) -> List[Trade]:
    def parse_order(order: dict) -> Trade:
        action = order["action"]

        if action == "buy":
            return Trade.buy(order)
        elif action == "sell":
            return Trade.sell(order)
        else:
            raise ValueError(f"Invalid action {action}")

    def parse_call(call) -> List[Trade]:
        arguments = json.loads(call.function.arguments)

        trades = [parse_order(order) for order in arguments["orders"]]
        return trades

    tool_spec = {
        "type": "function",
        "function": {
            "name": "trade",
            "description": "Execute orders to manage the portfolio. Orders will be executed immediately at the stated prices.",
            "parameters": {
                "type": "object",
                "properties": {
                    "orders": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["buy", "sell"]},
                                "asset": {"type": "string"},
                                "amount": {
                                    "type": "number",
                                    "description": "Amount of asset to buy or sell.",
                                },
                            },
                            "required": ["action", "asset", "amount"],
                        },
                    },
                },
            },
        },
    }

    response = completion(
        model_name,
        [
            {
                "role": "system",
                "content": """You are an expert asset manager, managing a portfolio.

                Always use the `trade` function. Make sure that you call it correctly. For example, the following is a valid call:
                ```
                trade({
                    "orders": [
                        {"action": "buy", "asset": "BTC", "amount": 0.1},
                        {"action": "sell", "asset": "ETH", "amount": 0.2}
                    ]
                })
                ```

                If there are no trades to make, call `trade` with an empty array:
                ```
                trade({ "orders": [] })
                ```
            """,
            },
            {
                "role": "user",
                "content": """Manage the portfolio.

            Don't jabber.

            This is the current market data:
            ```
            {market_data}
            ```

            Your portfolio is as follows:
            ```
            {portfolio}
            ```
            """.replace(
                    "{market_data}", "BTC: 64,000 USD\nETH: 3,500 USD"
                ).replace(
                    "{portfolio}", "USD: 1000, BTC: 0.1, ETH: 0.2"
                ),
            },
        ],
        tools=[tool_spec],
        tool_choice={
            "type": "function",
            "function": {"name": tool_spec["function"]["name"]},  # type: ignore
        },
    )

    calls = response.choices[0].message.tool_calls
    trades = [trade for call in calls for trade in parse_call(call)]
    return trades


@pytest.mark.parametrize(
    "model", ["claude-3-haiku-20240307", "anthropic.claude-3-haiku-20240307-v1:0"]
)
def test_function_call_parsing(model):
    trades = trade(model)
    print([trade.order for trade in trades])
