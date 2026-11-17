import json
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server


def test_unknown_model_provider_404_surfaces_to_client_as_404(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses", request.target
        body: Final = json.loads(request.body)
        assert body["model"] == "non-existent-model"
        return Reply(
            status=404,
            body=json.dumps(
                {"error": {"message": "model not found", "type": "invalid_request_error", "code": "404"}}
            ).encode(),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/non-existent-model", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request("POST", "/v1/responses", {"model": model, "input": "say hi"})
        assert response.status_code == 404, response.text
        assert len(wire.drain()) == 1


def test_provider_400_for_bad_temperature_surfaces_to_client_as_400(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/responses", request.target
        body: Final = json.loads(request.body)
        assert body["model"] == "gpt-4o"
        assert body["temperature"] == 2000
        return Reply(
            status=400,
            body=json.dumps(
                {"error": {"message": "temperature out of range", "type": "invalid_request_error", "code": "400"}}
            ).encode(),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-4o", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": model, "input": "say hi", "temperature": 2000}
        )
        assert response.status_code == 400, response.text
        assert len(wire.drain()) == 1


def test_cancel_invalid_response_id_surfaces_error_status(gateway: Gateway) -> None:
    response_id: Final = "invalid_response_id_12345"

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == f"/responses/{response_id}/cancel", request.target
        return Reply(
            status=404,
            body=json.dumps(
                {"error": {"message": "No such response", "type": "invalid_request_error", "code": "404"}}
            ).encode(),
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/responses/gpt-4o", api_base=wire.url, api_key="synthetic-openai-key"
        )
        response: Final = gateway.request("POST", f"/v1/responses/{response_id}/cancel", {"model": model})
        assert response.status_code == 404, response.text
        assert len(wire.drain()) == 1
