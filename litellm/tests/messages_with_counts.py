system_message_short = {
    "message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "count": 12
}

system_message = {
    "message": {
        "role": "system",
        "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
    },
    "count": 25
}

system_message_long = {
    "message": {
        "role": "system",
        "content": "Assistant helps the company employees with their healthcare plan questions, and questions about the employee handbook. Be brief in your answers.",
    },
    "count": 31
}

system_message_unicode = {
    "message": {
        "role": "system",
        "content": "á",
    },
    "count": 8
}

system_message_with_name = {
    "message": {
        "role": "system",
        "name": "example_user",
        "content": "New synergies will help drive top-line growth.",
    },
    "count": 20
}

user_message = {
    "message": {
        "role": "user",
        "content": "Hello, how are you?",
    },
    "count": 13
}

user_message_unicode = {
    "message": {
        "role": "user",
        "content": "á",
    },
    "count": 8
}

user_message_perf = {
    "message": {
        "role": "user",
        "content": "What happens in a performance review?",
    },
    "count": 14
}

assistant_message_perf = {
    "message": {
        "role": "assistant",
        "content": "During the performance review at Contoso Electronics, the supervisor will discuss the employee's performance over the past year and provide feedback on areas for improvement. They will also provide an opportunity for the employee to discuss their goals and objectives for the upcoming year. The review is a two-way dialogue between managers and employees, and employees will receive a written summary of their performance review which will include a rating of their performance, feedback, and goals and objectives for the upcoming year [employee_handbook-3.pdf].",
    },
    "count": 106
}

assistant_message_perf_short = {
    "message": {
        "role": "assistant",
        "content": "The supervisor will discuss the employee's performance and provide feedback on areas for improvement. They will also provide an opportunity for the employee to discuss their goals and objectives for the upcoming year. The review is a two-way dialogue between managers and employees, and employees will receive a written summary of their performance review which will include a rating of their performance, feedback, and goals for the upcoming year [employee_handbook-3.pdf].",
    },
    "count": 91
}

user_message_dresscode = {
    "message": {
        "role": "user",
        "content": "Is there a dress code?",
    },
    "count": 13
}

assistant_message_dresscode = {
    "message": {
        "role": "assistant",
        "content": "Yes, there is a dress code at Contoso Electronics. Look sharp! [employee_handbook-1.pdf]",
    },
    "count": 30
}

user_message_pm = {
    "message": {
        "role": "user",
        "content": "What does a Product Manager do?",
    },
    "count": 14
}

text_and_image_message = {
    "message": {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this picture:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z/C/HgAGgwJ/lK3Q6wAAAABJRU5ErkJggg==",
                    "detail": "high",
                },
            },
        ],
    },
    "count": 266
}



search_sources_toolchoice_auto = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "search_sources",
                "description": "Retrieve sources from the Azure AI Search index",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "Query string to retrieve documents from azure search eg: 'Health care plan'",
                        }
                    },
                    "required": ["search_query"],
                },
            },
        }
    ],
    "tool_choice": "auto",
    "count": 66,
}

search_sources_toolchoice_none = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "search_sources",
                "description": "Retrieve sources from the Azure AI Search index",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "Query string to retrieve documents from azure search eg: 'Health care plan'",
                        }
                    },
                    "required": ["search_query"],
                },
            },
        }
    ],
    "tool_choice": "none",
    "count": 67,
}

search_sources_toolchoice_name = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "search_sources",
                "description": "Retrieve sources from the Azure AI Search index",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "Query string to retrieve documents from azure search eg: 'Health care plan'",
                        }
                    },
                    "required": ["search_query"],
                },
            },
        }
    ],
    "tool_choice": {"type": "function", "function": {"name": "search_sources"}},
    "count": 75,
}

integer_enum = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "data_demonstration",
                "description": "This is the main function description",
                "parameters": {"type": "object", "properties": {"integer_enum": {"type": "integer", "enum": [-1, 1]}}},
            },
        }
    ],
    "tool_choice": "none",
    "count": 54,
}


integer_enum_tool_choice_name = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "data_demonstration",
                "description": "This is the main function description",
                "parameters": {"type": "object", "properties": {"integer_enum": {"type": "integer", "enum": [-1, 1]}}},
            },
        }
    ],
    "tool_choice": {
        "type": "function",
        "function": {"name": "data_demonstration"},
    },  # 4 tokens for "data_demonstration"
    "count": 64,
}

no_parameters = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "search_sources",
                "description": "Retrieve sources from the Azure AI Search index",
            },
        }
    ],
    "tool_choice": "auto",
    "count": 42,
}

no_parameters_tool_choice_name = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "search_sources",
                "description": "Retrieve sources from the Azure AI Search index",
            },
        }
    ],
    "tool_choice": {"type": "function", "function": {"name": "search_sources"}},  # 2 tokens for "search_sources"
    "count": 51,
}

no_parameter_description_or_required = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "search_sources",
                "description": "Retrieve sources from the Azure AI Search index",
                "parameters": {"type": "object", "properties": {"search_query": {"type": "string"}}},
            },
        }
    ],
    "tool_choice": "auto",
    "count": 49,
}

no_parameter_description = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "search_sources",
                "description": "Retrieve sources from the Azure AI Search index",
                "parameters": {
                    "type": "object",
                    "properties": {"search_query": {"type": "string"}},
                    "required": ["search_query"],
                },
            },
        }
    ],
    "tool_choice": "auto",
    "count": 49,
}

string_enum = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "summarize_order",
                "description": "Summarize the customer order request",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_name": {
                            "type": "string",
                            "description": "Product name ordered by customer",
                        },
                        "quantity": {
                            "type": "integer",
                            "description": "Quantity ordered by customer",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["meals", "days"],
                            "description": "unit of measurement of the customer order",
                        },
                    },
                    "required": ["product_name", "quantity", "unit"],
                },
            },
        }
    ],
    "tool_choice": "none",
    "count": 86,
}

inner_object = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
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
    ],
    "tool_choice": "none",
    "count": 65,  # counted 67, over by 2
}
"""
namespace functions {

// This is the main function description
type data_demonstration = (_: {
// The object data type as a property
object_1: {
  string1?: string,
},
}) => any;

} // namespace functions
"""

inner_object_with_enum_only = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
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
                            "properties": {"string_2a": {"type": "string", "enum": ["Happy", "Sad"]}},
                        }
                    },
                    "required": ["object_1"],
                },
            },
        }
    ],
    "tool_choice": "none",
    "count": 73,  # counted 74, over by 1
}
"""
namespace functions {

// This is the main function description
type data_demonstration = (_: {
// The object data type as a property
object_1: {
  string_2a?: "Happy" | "Sad",
},
}) => any;

} // namespace functions
"""

inner_object_with_enum = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
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
                                "string_2a": {"type": "string", "enum": ["Happy", "Sad"]},
                                "string_2b": {
                                    "type": "string",
                                    "description": "Description in a second object is lost",
                                },
                            },
                        }
                    },
                    "required": ["object_1"],
                },
            },
        }
    ],
    "tool_choice": "none",
    "count": 89,  # counted 92, over by 3
}
"""
namespace functions {

// This is the main function description
type data_demonstration = (_: {
// The object data type as a property
object_1: {
  string_2a?: "Happy" | "Sad",
  // Description in a second object is lost
  string_2b?: string,
},
}) => any;

} // namespace functions
"""

inner_object_and_string = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
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
                                "string_2a": {"type": "string", "enum": ["Happy", "Sad"]},
                                "string_2b": {
                                    "type": "string",
                                    "description": "Description in a second object is lost",
                                },
                            },
                        },
                        "string_1": {"type": "string", "description": "Not required gets a question mark"},
                    },
                    "required": ["object_1"],
                },
            },
        }
    ],
    "tool_choice": "none",
    "count": 103,  # counted 106, over by 3
}
"""
namespace functions {

// This is the main function description
type data_demonstration = (_: {
// The object data type as a property
object_1: {
  string_2a?: "Happy" | "Sad",
  // Description in a second object is lost
  string_2b?: string,
},
// Not required gets a question mark
string_1?: string,
}) => any;

} // namespace functions
"""

boolean = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "human_escalation",
                "description": "Check if user wants to escalate to a human",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "requires_escalation": {
                            "type": "boolean",
                            "description": "If user is showing signs of frustration or anger in the query. Also if the user says they want to talk to a real person and not a chat bot.",
                        }
                    },
                    "required": ["requires_escalation"],
                },
            },
        }
    ],
    "tool_choice": "none",
    "count": 89,  # over by 3
}

array = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_coordinates",
                "description": "Get the latitude and longitude of multiple mailing addresses",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "addresses": {
                            "type": "array",
                            "description": "The mailing addresses to be located",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["addresses"],
                },
            },
        }
    ],
    "tool_choice": "none",
    "count": 59,
}

null = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_null",
                "description": "Get the null value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "null_value": {
                            "type": "null",
                            "description": "The null value to be returned",
                        }
                    },
                    "required": ["null_value"],
                },
            },
        }
    ],
    "tool_choice": "none",
    "count": 55,
}

no_type = {
    "system_message": {
        "role": "system",
        "content": "You are a bot.",
    },
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_no_type",
                "description": "Get the no type value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "no_type_value": {
                            "description": "The no type value to be returned",
                        }
                    },
                    "required": ["no_type_value"],
                },
            },
        }
    ],
    "tool_choice": "none",
    "count": 59,
}

MESSAGES_TEXT = [
    system_message,
    system_message_short,
    system_message_long,
    system_message_unicode,
    system_message_with_name,
    user_message,
    user_message_unicode,
    user_message_perf,
    user_message_dresscode,
    user_message_pm,
    assistant_message_perf,
    assistant_message_perf_short,
    assistant_message_dresscode
]

MESSAGES_WITH_IMAGES = [
    text_and_image_message
]

MESSAGES_WITH_TOOLS = [
    inner_object,
    inner_object_and_string,
    inner_object_with_enum_only,
    inner_object_with_enum,
    search_sources_toolchoice_auto,
    search_sources_toolchoice_none,
    search_sources_toolchoice_name,
    integer_enum,
    integer_enum_tool_choice_name,
    no_parameters,
    no_parameters_tool_choice_name,
    no_parameter_description_or_required,
    no_parameter_description,
    string_enum,
    boolean,
    array,
    no_type,
    null,
]

