const messageSchema = {
  $defs: {
    Audio: {
      description:
        "Data about a previous audio response from the model.\n[Learn more](https://platform.openai.com/docs/guides/audio).",
      properties: { id: { title: "Id", type: "string" } },
      required: ["id"],
      title: "Audio",
      type: "object",
    },
    ChatCompletionAssistantMessageParam: {
      description: "Messages sent by the model in response to user messages.",
      properties: {
        role: { const: "assistant", title: "Role", type: "string" },
        audio: { anyOf: [{ $ref: "#/$defs/Audio" }, { type: "null" }] },
        content: {
          anyOf: [
            { type: "string" },
            {
              items: {
                anyOf: [
                  { $ref: "#/$defs/ChatCompletionContentPartTextParam" },
                  { $ref: "#/$defs/ChatCompletionContentPartRefusalParam" },
                ],
              },
              type: "array",
            },
            { type: "null" },
          ],
          title: "Content",
        },
        function_call: { anyOf: [{ $ref: "#/$defs/FunctionCall" }, { type: "null" }] },
        name: { title: "Name", type: "string" },
        refusal: { anyOf: [{ type: "string" }, { type: "null" }], title: "Refusal" },
        tool_calls: {
          items: {
            anyOf: [
              { $ref: "#/$defs/ChatCompletionMessageFunctionToolCallParam" },
              { $ref: "#/$defs/ChatCompletionMessageCustomToolCallParam" },
            ],
          },
          title: "Tool Calls",
          type: "array",
        },
      },
      required: ["role"],
      title: "ChatCompletionAssistantMessageParam",
      type: "object",
    },
    ChatCompletionContentPartImageParam: {
      description: "Learn about [image inputs](https://platform.openai.com/docs/guides/vision).",
      properties: {
        image_url: { $ref: "#/$defs/ImageURL" },
        type: { const: "image_url", title: "Type", type: "string" },
        prompt_cache_breakpoint: { $ref: "#/$defs/PromptCacheBreakpoint" },
      },
      required: ["image_url", "type"],
      title: "ChatCompletionContentPartImageParam",
      type: "object",
    },
    ChatCompletionContentPartInputAudioParam: {
      description: "Learn about [audio inputs](https://platform.openai.com/docs/guides/audio).",
      properties: {
        input_audio: { $ref: "#/$defs/InputAudio" },
        type: { const: "input_audio", title: "Type", type: "string" },
        prompt_cache_breakpoint: { $ref: "#/$defs/PromptCacheBreakpoint" },
      },
      required: ["input_audio", "type"],
      title: "ChatCompletionContentPartInputAudioParam",
      type: "object",
    },
    ChatCompletionContentPartRefusalParam: {
      properties: {
        refusal: { title: "Refusal", type: "string" },
        type: { const: "refusal", title: "Type", type: "string" },
      },
      required: ["refusal", "type"],
      title: "ChatCompletionContentPartRefusalParam",
      type: "object",
    },
    ChatCompletionContentPartTextParam: {
      description: "Learn about [text inputs](https://platform.openai.com/docs/guides/text-generation).",
      properties: {
        text: { title: "Text", type: "string" },
        type: { const: "text", title: "Type", type: "string" },
        prompt_cache_breakpoint: { $ref: "#/$defs/PromptCacheBreakpoint" },
      },
      required: ["text", "type"],
      title: "ChatCompletionContentPartTextParam",
      type: "object",
    },
    ChatCompletionDeveloperMessageParam: {
      description:
        "Developer-provided instructions that the model should follow, regardless of\nmessages sent by the user. With o1 models and newer, `developer` messages\nreplace the previous `system` messages.",
      properties: {
        content: {
          anyOf: [{ type: "string" }, { items: { $ref: "#/$defs/ChatCompletionContentPartTextParam" }, type: "array" }],
          title: "Content",
        },
        role: { const: "developer", title: "Role", type: "string" },
        name: { title: "Name", type: "string" },
      },
      required: ["content", "role"],
      title: "ChatCompletionDeveloperMessageParam",
      type: "object",
    },
    ChatCompletionFunctionMessageParam: {
      properties: {
        content: { anyOf: [{ type: "string" }, { type: "null" }], title: "Content" },
        name: { title: "Name", type: "string" },
        role: { const: "function", title: "Role", type: "string" },
      },
      required: ["content", "name", "role"],
      title: "ChatCompletionFunctionMessageParam",
      type: "object",
    },
    ChatCompletionMessageCustomToolCallParam: {
      description: "A call to a custom tool created by the model.",
      properties: {
        id: { title: "Id", type: "string" },
        custom: { $ref: "#/$defs/Custom" },
        type: { const: "custom", title: "Type", type: "string" },
      },
      required: ["id", "custom", "type"],
      title: "ChatCompletionMessageCustomToolCallParam",
      type: "object",
    },
    ChatCompletionMessageFunctionToolCallParam: {
      description: "A call to a function tool created by the model.",
      properties: {
        id: { title: "Id", type: "string" },
        function: { $ref: "#/$defs/Function" },
        type: { const: "function", title: "Type", type: "string" },
      },
      required: ["id", "function", "type"],
      title: "ChatCompletionMessageFunctionToolCallParam",
      type: "object",
    },
    ChatCompletionSystemMessageParam: {
      description:
        "Developer-provided instructions that the model should follow, regardless of\nmessages sent by the user. With o1 models and newer, use `developer` messages\nfor this purpose instead.",
      properties: {
        content: {
          anyOf: [{ type: "string" }, { items: { $ref: "#/$defs/ChatCompletionContentPartTextParam" }, type: "array" }],
          title: "Content",
        },
        role: { const: "system", title: "Role", type: "string" },
        name: { title: "Name", type: "string" },
      },
      required: ["content", "role"],
      title: "ChatCompletionSystemMessageParam",
      type: "object",
    },
    ChatCompletionToolMessageParam: {
      properties: {
        content: {
          anyOf: [{ type: "string" }, { items: { $ref: "#/$defs/ChatCompletionContentPartTextParam" }, type: "array" }],
          title: "Content",
        },
        role: { const: "tool", title: "Role", type: "string" },
        tool_call_id: { title: "Tool Call Id", type: "string" },
      },
      required: ["content", "role", "tool_call_id"],
      title: "ChatCompletionToolMessageParam",
      type: "object",
    },
    ChatCompletionUserMessageParam: {
      description: "Messages sent by an end user, containing prompts or additional context\ninformation.",
      properties: {
        content: {
          anyOf: [
            { type: "string" },
            {
              items: {
                anyOf: [
                  { $ref: "#/$defs/ChatCompletionContentPartTextParam" },
                  { $ref: "#/$defs/ChatCompletionContentPartImageParam" },
                  { $ref: "#/$defs/ChatCompletionContentPartInputAudioParam" },
                  { $ref: "#/$defs/File" },
                ],
              },
              type: "array",
            },
          ],
          title: "Content",
        },
        role: { const: "user", title: "Role", type: "string" },
        name: { title: "Name", type: "string" },
      },
      required: ["content", "role"],
      title: "ChatCompletionUserMessageParam",
      type: "object",
    },
    Custom: {
      description: "The custom tool that the model called.",
      properties: { input: { title: "Input", type: "string" }, name: { title: "Name", type: "string" } },
      required: ["input", "name"],
      title: "Custom",
      type: "object",
    },
    File: {
      description: "Learn about [file inputs](https://platform.openai.com/docs/guides/text) for text generation.",
      properties: {
        file: { $ref: "#/$defs/FileFile" },
        type: { const: "file", title: "Type", type: "string" },
        prompt_cache_breakpoint: { $ref: "#/$defs/FilePromptCacheBreakpoint" },
      },
      required: ["file", "type"],
      title: "File",
      type: "object",
    },
    FileFile: {
      properties: {
        file_data: { title: "File Data", type: "string" },
        file_id: { title: "File Id", type: "string" },
        filename: { title: "Filename", type: "string" },
      },
      title: "FileFile",
      type: "object",
    },
    FilePromptCacheBreakpoint: {
      description:
        "Marks the exact end of a reusable prompt prefix.\n\nThe breakpoint inherits its TTL from the request's `prompt_cache_options.ttl`; the boundary is not rounded to a token block.",
      properties: { mode: { const: "explicit", title: "Mode", type: "string" } },
      required: ["mode"],
      title: "FilePromptCacheBreakpoint",
      type: "object",
    },
    Function: {
      description: "The function that the model called.",
      properties: { arguments: { title: "Arguments", type: "string" }, name: { title: "Name", type: "string" } },
      required: ["arguments", "name"],
      title: "Function",
      type: "object",
    },
    FunctionCall: {
      description:
        "Deprecated and replaced by `tool_calls`.\n\nThe name and arguments of a function that should be called, as generated by the model.",
      properties: { arguments: { title: "Arguments", type: "string" }, name: { title: "Name", type: "string" } },
      required: ["arguments", "name"],
      title: "FunctionCall",
      type: "object",
    },
    ImageURL: {
      properties: {
        url: { title: "Url", type: "string" },
        detail: { enum: ["auto", "low", "high"], title: "Detail", type: "string" },
      },
      required: ["url"],
      title: "ImageURL",
      type: "object",
    },
    InputAudio: {
      properties: {
        data: { title: "Data", type: "string" },
        format: { enum: ["wav", "mp3"], title: "Format", type: "string" },
      },
      required: ["data", "format"],
      title: "InputAudio",
      type: "object",
    },
    PromptCacheBreakpoint: {
      description:
        "Marks the exact end of a reusable prompt prefix.\n\nThe breakpoint inherits its TTL from the request's `prompt_cache_options.ttl`; the boundary is not rounded to a token block.",
      properties: { mode: { const: "explicit", title: "Mode", type: "string" } },
      required: ["mode"],
      title: "PromptCacheBreakpoint",
      type: "object",
    },
  },
  items: {
    anyOf: [
      { $ref: "#/$defs/ChatCompletionDeveloperMessageParam" },
      { $ref: "#/$defs/ChatCompletionSystemMessageParam" },
      { $ref: "#/$defs/ChatCompletionUserMessageParam" },
      { $ref: "#/$defs/ChatCompletionAssistantMessageParam" },
      { $ref: "#/$defs/ChatCompletionToolMessageParam" },
      { $ref: "#/$defs/ChatCompletionFunctionMessageParam" },
    ],
  },
  type: "array",
};

const draft = {
  name: "email-reply",
  workflow_id: "fixture-email-reply",
  team_id: "fixture-team",
  etag: '"098b942710af75cd3421f9465ffc114ec6a7e2be034076c7a593fcc1b44df402"',
  digest: "098b942710af75cd3421f9465ffc114ec6a7e2be034076c7a593fcc1b44df402",
  bundle: {
    contract_version: "1",
    name: "email-reply",
    timeout_seconds: 300.0,
    input_schema: messageSchema,
    output: { from: "node_3", output: "result" },
    nodes: [
      {
        id: "node_1",
        timeout_seconds: 60.0,
        retry: { max_attempts: 1, backoff_seconds: 0.0 },
        type: "chat_completion",
        model: "support-model",
        messages: { from: "$input", output: "messages" },
        temperature: 0.4,
      },
      {
        id: "node_2",
        timeout_seconds: 15,
        retry: { max_attempts: 2, backoff_seconds: 0.5 },
        type: "python",
        program: {
          contract_version: "1",
          source:
            "def prepare_followup(text: str, instruction: str) -> list[ChatMessage]:\n    return [{'role': 'user', 'content': instruction + '\\n\\n' + text}]\n",
          function_name: "prepare_followup",
          parameter_names: ["text", "instruction"],
          input_schema: {
            additionalProperties: false,
            properties: {
              text: { title: "Text", type: "string" },
              instruction: { title: "Instruction", type: "string" },
            },
            required: ["text", "instruction"],
            title: "Arguments",
            type: "object",
          },
          output_schema: messageSchema,
          source_sha256: "81685c2f4773b08d0b26d6809b996b8c5ccd159a80f5091e2d5744169ee5cb51",
        },
        inputs: {
          text: { from: "node_1", output: "result" },
          instruction: { value: "Review this draft for clarity and accuracy" },
        },
      },
      {
        id: "node_3",
        timeout_seconds: 60.0,
        retry: { max_attempts: 1, backoff_seconds: 0.0 },
        type: "chat_completion",
        model: "review-model",
        messages: { from: "node_2", output: "result" },
        max_completion_tokens: 512,
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "Review",
            strict: true,
            schema: {
              properties: {
                summary: { title: "Summary", type: "string" },
                approved: { title: "Approved", type: "boolean" },
              },
              required: ["summary", "approved"],
              title: "Review",
              type: "object",
            },
          },
        },
      },
    ],
  },
  published_snapshot_id: null,
};

export default draft;
