from litellm.llms.vertex_ai.agent_engine.transformation import VertexAgentEngineConfig

MULTI_TURN = [
    {"role": "user", "content": "Remember the number 42"},
    {"role": "assistant", "content": "Noted."},
    {"role": "user", "content": "Which number?"},
]


def _transform(messages, optional_params=None, litellm_params=None) -> dict:
    config = VertexAgentEngineConfig()
    return config.transform_request(
        model="agent_engine/123456789",
        messages=messages,
        optional_params=optional_params if optional_params is not None else {"user_id": "u1"},
        litellm_params=litellm_params if litellm_params is not None else {},
        headers={},
    )


class TestPromptWithoutSession:
    def test_history_travels_in_the_prompt(self):
        message = _transform(MULTI_TURN)["input"]["message"]

        assert "Remember the number 42" in message
        assert "Noted." in message
        assert "Which number?" in message

    def test_roles_are_labelled_so_turns_stay_distinguishable(self):
        message = _transform(MULTI_TURN)["input"]["message"]

        assert message.startswith("user: Remember the number 42")
        assert "assistant: Noted." in message

    def test_a_single_message_is_sent_verbatim(self):
        message = _transform([{"role": "user", "content": "Hello"}])["input"]["message"]

        assert message == "Hello"

    def test_list_content_is_flattened(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "first"}]},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ]

        message = _transform(messages)["input"]["message"]

        assert message == "user: first\n\nuser: second"


class TestPromptWithSession:
    def test_only_the_new_message_is_sent(self):
        result = _transform(MULTI_TURN, optional_params={"user_id": "u1", "session_id": "s-1"})

        assert result["input"]["message"] == "Which number?"
        assert result["input"]["session_id"] == "s-1"


class TestSessionResolution:
    def test_no_session_key_is_sent_when_none_is_given(self):
        assert "session_id" not in _transform(MULTI_TURN)["input"]

    def test_the_proxy_session_does_not_become_an_agent_session(self):
        """An Agent Engine session is keyed by user too, and `_get_user_id` invents a new
        one per request, so adopting the proxy's session would send only the newest message
        to a session that can never be found again."""
        result = _transform(MULTI_TURN, litellm_params={"litellm_session_id": "s-proxy"})

        assert "session_id" not in result["input"]
        assert "Remember the number 42" in result["input"]["message"]
