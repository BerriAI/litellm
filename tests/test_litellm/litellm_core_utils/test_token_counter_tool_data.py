SHORT = [
    {"role": "system", "content": "System prompt"},
    {
        "tool_call_id": "call_E0lOb1h6qtmflUyok4L06TgY",
        "role": "tool",
        "name": "SearchInternet",
        "content": "tool content",
    },
]

CONTENT_AND_TOOL_CALL = [
    {"role": "system", "content": "System prompt"},
    {
        "role": "assistant",
        "content": "..",
        "tool_calls": [
            {
                "id": "toolu_01NZjJ3e6fzhhbYYcb5k1k4d",
                "type": "function",
                "function": {
                    "name": "str_replace_editor",
                    "arguments": '{"command": "view","path": "/workspace"}',
                },
            }
        ],
    },
]

_OPENHANDS_SYSTEM_MESSAGE = {
    "content": [
        {
            "type": "text",
            "text": "You are OpenHands agent, a helpful AI assistant that can "
            "interact with a computer to solve tasks.\n\n<ROLE>\nYour primary "
            "role is to assist users by executing commands, modifying code, and "
            "solving technical problems effectively. You should be thorough, "
            "methodical, and prioritize quality over speed.\n* If the user asks a "
            "question, like 'why is X happening', don’t try to fix the problem. ",
        }
    ],
    "role": "system",
}

SYSTEM_LONG = [
    _OPENHANDS_SYSTEM_MESSAGE,
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "toolu_01NZjJ3e6fzhhbYYcb5k1k4d",
                "type": "function",
                "function": {
                    "name": "str_replace_editor",
                    "arguments": '{"command": "view","path": "/workspace"}',
                },
            }
        ],
    },
]

TOOL_CALL_CONTENT_ARRAY = [
    _OPENHANDS_SYSTEM_MESSAGE,
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "very short"}],
        "tool_calls": [
            {
                "id": "toolu_01NZjJ3e6fzhhbYYcb5k1k4d",
                "type": "function",
                "function": {
                    "name": "str_replace_editor",
                    "arguments": '{"command": "view","path": "/workspace"}',
                },
            }
        ],
    },
]

USER_MESSAGES = [
    {"role": "system", "content": "System prompt"},
    {"content": [{"type": "text", "text": "2025-04-25 (UTC)"}], "role": "user"},
    {
        "role": "system",
        "content": "System prompt long " + ("long" * 100),
    },
    {
        "role": "system",
        "content": ["System prompt long " + ("long" * 100)],
    },
    {
        "role": "system",
        "content": [{"type": "text", "text": "System prompt long " + ("long" * 100)}],
    },
]
USER_MESSAGES_IDS = ["system", "user-arr", "long", "long-arr", "long-arr-obj"]

TOOL_CALL_MESSAGES = [
    {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "x",
                "type": "function",
                "function": {
                    "arguments": '{"arg":"value"}',
                    "name": "foo",
                },
            }
        ],
    },
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_E0lOb1h6qtmflUyok4L06TgY",
                "function": {
                    "arguments": '{"query":"search query","domain":"google.ca","gl":"ca","hl":"en"}',
                    "name": "SearchInternet",
                },
                "type": "function",
            }
        ],
    },
    {"role": "user", "content": "Hi"},
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "very short"}],
        "tool_calls": [
            {
                "id": "toolu_01NZjJ3e6fzhhbYYcb5k1k4d",
                "type": "function",
                "function": {
                    "name": "str_replace_editor",
                    "arguments": '{"command": "view","path": "/workspace"}',
                },
            }
        ],
    },
]

TOOL_CALL_MESSAGE_ids = ["t_short", "t_normal", "usermsg", "t_content_array"]


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "data_demonstration",
            "description": "This is the main function description",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_1": {
                        "type": "object",
                        "description": "The object data type as a property",
                        "properties": {
                            "string1": {"type": "string"},
                        },
                    }
                },
                "required": ["object_1"],
            },
        },
    }
]
