from litellm.proxy.gateway.mcp.foundation.errors import GatewayError, reason


def test_construct_each_arm_carries_its_payload():
    assert GatewayError(db_unavailable="x").tag == "db_unavailable"
    assert GatewayError(unauthorized="x").tag == "unauthorized"
    assert GatewayError(invalid_input="x").tag == "invalid_input"
    assert GatewayError(not_implemented="boom").not_implemented == "boom"


def test_reason_returns_the_active_arm_payload():
    assert reason(GatewayError(db_unavailable="db is down")) == "db is down"
    assert reason(GatewayError(unauthorized="nope")) == "nope"
    assert reason(GatewayError(invalid_input="bad name")) == "bad name"
    assert reason(GatewayError(not_implemented="later")) == "later"
