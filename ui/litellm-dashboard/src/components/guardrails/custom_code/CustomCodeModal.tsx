import React, { useState, useRef, useEffect } from "react";
import { Modal, Select, Switch, Collapse, Input, Divider } from "antd";
import { Button, TextInput } from "@tremor/react";
import {
  CodeOutlined,
  PlayCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  CaretRightOutlined,
  SaveOutlined,
  UsergroupAddOutlined,
  ExportOutlined,
} from "@ant-design/icons";
import { createGuardrailCall, updateGuardrailCall, testCustomCodeGuardrail } from "../../networking";
import NotificationsManager from "../../molecules/notifications_manager";

const { Panel } = Collapse;
const { TextArea } = Input;

// Code templates
const CODE_TEMPLATES = {
  empty: {
    name: "Empty Template",
    code: `async def apply_guardrail(inputs, request_data, input_type):
    # inputs: {texts, images, tools, tool_calls, structured_messages, model}
    # request_data: {model, user_id, team_id, end_user_id, metadata}
    # input_type: "request" or "response"
    return allow()`,
  },
  blockSSN: {
    name: "Block SSN",
    code: `def apply_guardrail(inputs, request_data, input_type):
    for text in inputs["texts"]:
        if regex_match(text, r"\\d{3}-\\d{2}-\\d{4}"):
            return block("SSN detected")
    return allow()`,
  },
  redactEmail: {
    name: "Redact Emails",
    code: `def apply_guardrail(inputs, request_data, input_type):
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
    modified = []
    for text in inputs["texts"]:
        modified.append(regex_replace(text, pattern, "[EMAIL REDACTED]"))
    return modify(texts=modified)`,
  },
  blockSQL: {
    name: "Block SQL Injection",
    code: `def apply_guardrail(inputs, request_data, input_type):
    if input_type != "request":
        return allow()
    for text in inputs["texts"]:
        if contains_code_language(text, ["sql"]):
            return block("SQL code not allowed")
    return allow()`,
  },
  validateJSON: {
    name: "Validate JSON",
    code: `def apply_guardrail(inputs, request_data, input_type):
    if input_type != "response":
        return allow()
    
    schema = {"type": "object", "required": ["name", "value"]}
    
    for text in inputs["texts"]:
        obj = json_parse(text)
        if obj is None:
            return block("Invalid JSON response")
        if not json_schema_valid(obj, schema):
            return block("Response missing required fields")
    return allow()`,
  },
  externalAPI: {
    name: "External API Check (async)",
    code: `async def apply_guardrail(inputs, request_data, input_type):
    # Call an external moderation API (async for non-blocking)
    for text in inputs["texts"]:
        response = await http_post(
            "https://api.example.com/moderate",
            body={"text": text, "user_id": request_data["user_id"]},
            headers={"Authorization": "Bearer YOUR_API_KEY"},
            timeout=10
        )
        
        if not response["success"]:
            # API call failed, allow by default or block
            return allow()
        
        if response["body"].get("flagged"):
            return block(response["body"].get("reason", "Content flagged"))
    
    return allow()`,
  },
};


// Available primitives organized by category
const PRIMITIVES = {
  "Return Values": [
    { name: "allow()", desc: "Let request/response through" },
    { name: "block(reason)", desc: "Reject with message" },
    { name: "modify(texts=[], images=[], tool_calls=[])", desc: "Transform content" },
  ],
  "HTTP Requests (async)": [
    { name: "await http_request(url, method, headers, body)", desc: "Make async HTTP request" },
    { name: "await http_get(url, headers)", desc: "Async GET request" },
    { name: "await http_post(url, body, headers)", desc: "Async POST request" },
  ],
  "Regex Functions": [
    { name: "regex_match(text, pattern)", desc: "Returns True if pattern found" },
    { name: "regex_replace(text, pattern, replacement)", desc: "Replace all matches" },
    { name: "regex_find_all(text, pattern)", desc: "Return list of matches" },
  ],
  "JSON Functions": [
    { name: "json_parse(text)", desc: "Parse JSON string, returns None on error" },
    { name: "json_stringify(obj)", desc: "Convert to JSON string" },
    { name: "json_schema_valid(obj, schema)", desc: "Validate against JSON schema" },
  ],
  "URL Functions": [
    { name: "extract_urls(text)", desc: "Extract all URLs from text" },
    { name: "is_valid_url(url)", desc: "Check if URL is valid" },
    { name: "all_urls_valid(text)", desc: "Check all URLs in text are valid" },
  ],
  "Code Detection": [
    { name: "detect_code(text)", desc: "Returns True if code detected" },
    { name: "detect_code_languages(text)", desc: "Returns list of detected languages" },
    { name: 'contains_code_language(text, ["sql"])', desc: "Check for specific languages" },
  ],
  "Text Utilities": [
    { name: "contains(text, substring)", desc: "Check if substring exists" },
    { name: "contains_any(text, [substr1, substr2])", desc: "Check if any substring exists" },
    { name: "word_count(text)", desc: "Count words" },
    { name: "char_count(text)", desc: "Count characters" },
    { name: "lower(text) / upper(text) / trim(text)", desc: "String transforms" },
  ],
};

const MODE_OPTIONS = [
  { value: "pre_call", label: "pre_call (Request)" },
  { value: "post_call", label: "post_call (Response)" },
  { value: "during_call", label: "during_call (Parallel)" },
  { value: "logging_only", label: "logging_only" },
  { value: "pre_mcp_call", label: "pre_mcp_call (Before MCP Tool Call)" },
  { value: "post_mcp_call", label: "post_mcp_call (After MCP Tool Call)" },
  { value: "during_mcp_call", label: "during_mcp_call (During MCP Tool Call)" },
];

// Data for editing an existing guardrail
export interface EditGuardrailData {
  guardrail_id: string;
  guardrail_name: string;
  litellm_params: {
    mode?: string | string[];
    default_on?: boolean;
    custom_code?: string;
    [key: string]: any;
  };
}

interface CustomCodeModalProps {
  visible: boolean;
  onClose: () => void;
  onSuccess: () => void;
  accessToken: string | null;
  /** If provided, the modal will be in edit mode */
  editData?: EditGuardrailData | null;
}

const CustomCodeModal: React.FC<CustomCodeModalProps> = ({
  visible,
  onClose,
  onSuccess,
  accessToken,
  editData,
}) => {
  const isEditMode = !!editData;
  const [guardrailName, setGuardrailName] = useState("");
  const [mode, setMode] = useState<string[]>(["pre_call"]);
  const [defaultOn, setDefaultOn] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<string>("empty");
  const [code, setCode] = useState(CODE_TEMPLATES.empty.code);
  const [isSaving, setIsSaving] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [testExpanded, setTestExpanded] = useState(false);
  
  // Test input examples for pre_call and post_call
  const TEST_INPUT_EXAMPLES = {
    pre_call: {
      name: "Pre-call (Request)",
      data: {
        texts: ["Hello, my SSN is 123-45-6789"],
        images: [],
        tools: [
          {
            type: "function",
            function: {
              name: "get_weather",
              description: "Get the current weather in a location",
              parameters: {
                type: "object",
                properties: {
                  location: { type: "string", description: "City name" }
                },
                required: ["location"]
              }
            }
          }
        ],
        tool_calls: [],
        structured_messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: "Hello, my SSN is 123-45-6789" }
        ],
        model: "gpt-4"
      }
    },
    post_call: {
      name: "Post-call (Response)",
      data: {
        texts: ["The weather in San Francisco is 72°F and sunny."],
        images: [],
        tools: [],
        tool_calls: [
          {
            id: "call_abc123",
            type: "function",
            function: {
              name: "get_weather",
              arguments: "{\"location\": \"San Francisco\"}"
            }
          }
        ],
        structured_messages: [],
        model: "gpt-4"
      }
    },
    pre_mcp_call: {
      name: "Pre MCP (MCP tool as OpenAI tool)",
      data: {
        texts: [
          "Tool: read_wiki_structure\nArguments: {\"repoName\": \"BerriAI/litellm\"}"
        ],
        images: [],
        tools: [
          {
            type: "function",
            function: {
              name: "read_wiki_structure",
              description: "Read the structure of a GitHub repository (MCP tool passed as OpenAI tool)",
              parameters: {
                type: "object",
                properties: {
                  repoName: { type: "string", description: "Repository name, e.g. BerriAI/litellm" }
                },
                required: ["repoName"]
              }
            }
          }
        ],
        tool_calls: [
          {
            id: "call_mcp_001",
            type: "function",
            function: {
              name: "read_wiki_structure",
              arguments: "{\"repoName\": \"BerriAI/litellm\"}"
            }
          }
        ],
        structured_messages: [
          { role: "user", content: "Tool: read_wiki_structure\nArguments: {\"repoName\": \"BerriAI/litellm\"}" }
        ],
        model: "mcp-tool-call"
      }
    }
  };
  
  const [testInput, setTestInput] = useState(JSON.stringify(TEST_INPUT_EXAMPLES.pre_call.data, null, 2));
  const [testResult, setTestResult] = useState<any>(null);
  const [copiedPrimitive, setCopiedPrimitive] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Handle template change
  const handleTemplateChange = (templateKey: string) => {
    setSelectedTemplate(templateKey);
    
    // Check if it's a standard template
    setCode(CODE_TEMPLATES[templateKey as keyof typeof CODE_TEMPLATES].code);
  };

  // Normalize mode from API (string or string[]) to string[]
  const normalizeMode = (m: string | string[] | undefined): string[] => {
    if (m === undefined || m === null) return ["pre_call"];
    if (Array.isArray(m)) return m.length ? m : ["pre_call"];
    return [m];
  };

  // Reset form when modal opens or editData changes
  useEffect(() => {
    if (visible) {
      if (editData) {
        // Edit mode: populate with existing data
        setGuardrailName(editData.guardrail_name || "");
        setMode(normalizeMode(editData.litellm_params?.mode));
        setDefaultOn(editData.litellm_params?.default_on || false);
        setCode(editData.litellm_params?.custom_code || CODE_TEMPLATES.empty.code);
        setSelectedTemplate(""); // No template selected in edit mode
      } else {
        // Create mode: reset to defaults
        setGuardrailName("");
        setMode(["pre_call"]);
        setDefaultOn(false);
        setSelectedTemplate("empty");
        setCode(CODE_TEMPLATES.empty.code);
      }
      setTestResult(null);
      setTestExpanded(false);
    }
  }, [visible, editData]);

  // Copy primitive to clipboard
  const copyPrimitive = async (primitive: string) => {
    try {
      await navigator.clipboard.writeText(primitive);
      setCopiedPrimitive(primitive);
      setTimeout(() => setCopiedPrimitive(null), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  // Handle tab key in textarea
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Tab") {
      e.preventDefault();
      const textarea = e.currentTarget;
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const newValue = code.substring(0, start) + "    " + code.substring(end);
      setCode(newValue);
      setTimeout(() => {
        textarea.selectionStart = textarea.selectionEnd = start + 4;
      }, 0);
    }
  };

  // Save guardrail (create or update)
  const handleSave = async () => {
    if (!guardrailName.trim()) {
      NotificationsManager.fromBackend("Please enter a guardrail name");
      return;
    }
    if (!code.trim()) {
      NotificationsManager.fromBackend("Please enter custom code");
      return;
    }
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
      return;
    }

    setIsSaving(true);
    try {
      if (isEditMode && editData) {
        // Update existing guardrail
        const updateData: any = {
          litellm_params: {
            custom_code: code,
          },
        };

        // Only include changed fields
        if (guardrailName !== editData.guardrail_name) {
          updateData.guardrail_name = guardrailName;
        }
        const existingMode = normalizeMode(editData.litellm_params?.mode);
        const modeChanged =
          mode.length !== existingMode.length ||
          mode.some((m, i) => m !== existingMode[i]);
        if (modeChanged) {
          updateData.litellm_params.mode = mode;
        }
        if (defaultOn !== editData.litellm_params?.default_on) {
          updateData.litellm_params.default_on = defaultOn;
        }

        await updateGuardrailCall(accessToken, editData.guardrail_id, updateData);
        NotificationsManager.success("Custom code guardrail updated successfully");
      } else {
        // Create new guardrail
        const guardrailData = {
          guardrail_name: guardrailName,
          litellm_params: {
            guardrail: "custom_code",
            mode: mode,
            default_on: defaultOn,
            custom_code: code,
          },
          guardrail_info: {},
        };

        await createGuardrailCall(accessToken, guardrailData);
        NotificationsManager.success("Custom code guardrail created successfully");
      }
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to save guardrail:", error);
      NotificationsManager.fromBackend(
        `Failed to ${isEditMode ? "update" : "create"} guardrail: ` + (error instanceof Error ? error.message : String(error))
      );
    } finally {
      setIsSaving(false);
    }
  };

  // Test guardrail using backend endpoint
  const handleTest = async () => {
    if (!accessToken) {
      setTestResult({ error: "No access token available" });
      return;
    }

    setIsTesting(true);
    setTestResult(null);

    try {
      // Parse test input JSON
      let parsedInput;
      try {
        parsedInput = JSON.parse(testInput);
      } catch (e) {
        setTestResult({ error: "Invalid test input JSON" });
        setIsTesting(false);
        return;
      }

      // Ensure texts array exists
      if (!parsedInput.texts) {
        parsedInput.texts = [];
      }

      // Use first request-like or response-like mode for test input_type
      const requestModes = ["pre_call", "pre_mcp_call"];
      const responseModes = ["post_call", "post_mcp_call"];
      const testInputType: "request" | "response" =
        mode.some((m) => requestModes.includes(m))
          ? "request"
          : mode.some((m) => responseModes.includes(m))
            ? "response"
            : "request";

      const response = await testCustomCodeGuardrail(accessToken, {
        custom_code: code,
        test_input: parsedInput,
        input_type: testInputType,
        request_data: {
          model: "test-model",
          metadata: {},
        },
      });

      if (response.success && response.result) {
        setTestResult(response.result);
      } else if (response.error) {
        setTestResult({
          error: response.error,
          error_type: response.error_type,
        });
      } else {
        setTestResult({ error: "Unknown error occurred" });
      }
    } catch (error) {
      console.error("Failed to test custom code:", error);
      setTestResult({
        error: error instanceof Error ? error.message : "Failed to test custom code",
      });
    } finally {
      setIsTesting(false);
    }
  };

  const lineCount = code.split("\n").length;

  return (
    <Modal
      open={visible}
      onCancel={onClose}
      footer={null}
      width={1400}
      className="custom-code-modal"
      closable={true}
      destroyOnClose
    >
      <div className="flex flex-col h-[80vh]">
        {/* Header */}
        <div className="pb-4 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">
            {isEditMode ? "Edit Custom Guardrail" : "Create Custom Guardrail"}
          </h2>
          <p className="text-sm text-gray-500 mt-1">Define custom logic using Python-like syntax</p>
        </div>

        {/* Top Controls */}
        <div className="flex items-center gap-4 py-4 border-b border-gray-100">
          <div className="flex-1 max-w-[200px]">
            <label className="block text-xs font-medium text-gray-600 mb-1">Guardrail Name</label>
            <TextInput
              value={guardrailName}
              onValueChange={setGuardrailName}
              placeholder="e.g., block-pii-custom"
            />
          </div>
          <div className="w-[280px]">
            <label className="block text-xs font-medium text-gray-600 mb-1">Mode (can select multiple)</label>
            <Select
              mode="multiple"
              value={mode}
              onChange={setMode}
              options={MODE_OPTIONS}
              className="w-full"
              size="middle"
              placeholder="Select modes"
            />
          </div>
          <div className="w-[180px]">
            <label className="block text-xs font-medium text-gray-600 mb-1">Template</label>
            <Select
              value={selectedTemplate}
              onChange={handleTemplateChange}
              className="w-full"
              size="middle"
              dropdownRender={(menu) => (
                <>
                  {menu}
                  <Divider style={{ margin: '8px 0' }} />
                  <div
                    style={{
                      padding: '8px 12px',
                      cursor: 'pointer',
                      color: '#1890ff',
                      fontSize: '12px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '4px',
                    }}
                    onClick={(e) => {
                      e.preventDefault();
                      window.open('https://models.litellm.ai/guardrails', '_blank');
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = '#f0f0f0';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }}
                  >
                    <UsergroupAddOutlined />
                    <span>Browse Community templates</span>
                    <ExportOutlined style={{ fontSize: '10px' }} />
                  </div>
                </>
              )}
            >
              <Select.OptGroup label="STANDARD">
                {Object.entries(CODE_TEMPLATES).map(([key, template]) => (
                  <Select.Option key={key} value={key}>
                    {template.name}
                  </Select.Option>
                ))}
              </Select.OptGroup>
            </Select>
          </div>
          <div className="flex items-center gap-2 pt-5">
            <span className="text-sm text-gray-600">Default On</span>
            <Switch checked={defaultOn} onChange={setDefaultOn} />
          </div>
        </div>

        {/* Main Content */}
        <div className="flex flex-1 overflow-hidden mt-4 gap-6">
          {/* Code Editor */}
          <div className="flex-[2] flex flex-col min-w-0 overflow-y-auto">
            <div className="flex items-center justify-between mb-2 flex-shrink-0">
              <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Python Logic</span>
              <span className="text-xs text-gray-400">Restricted environment (no imports)</span>
            </div>
            <div className="relative rounded-lg overflow-hidden border border-gray-700 bg-[#1e1e1e] flex-shrink-0" style={{ minHeight: "300px", maxHeight: "400px" }}>
              {/* Line numbers */}
              <div 
                className="absolute left-0 top-0 bottom-0 w-12 bg-[#1e1e1e] border-r border-gray-700 text-right pr-3 pt-3 select-none overflow-hidden"
                style={{ fontFamily: "'Fira Code', 'Monaco', 'Consolas', monospace", fontSize: "14px", lineHeight: "1.6" }}
              >
                {Array.from({ length: Math.max(lineCount, 20) }, (_, i) => (
                  <div key={i + 1} className="text-gray-500 h-[22.4px]">{i + 1}</div>
                ))}
              </div>
              {/* Code textarea */}
              <textarea
                ref={textareaRef}
                value={code}
                onChange={(e) => setCode(e.target.value)}
                onKeyDown={handleKeyDown}
                spellCheck={false}
                className="w-full h-full pl-14 pr-4 pt-3 pb-3 resize-none focus:outline-none bg-transparent text-gray-200"
                style={{ fontFamily: "'Fira Code', 'Monaco', 'Consolas', monospace", fontSize: "14px", lineHeight: "1.6", tabSize: 4 }}
              />
            </div>

            {/* Test Section */}
            <Collapse
              activeKey={testExpanded ? ["test"] : []}
              onChange={(keys) => setTestExpanded(keys.includes("test"))}
              className="mt-3 bg-white border border-gray-200 rounded-lg flex-shrink-0"
              expandIcon={({ isActive }) => <CaretRightOutlined rotate={isActive ? 90 : 0} />}
            >
              <Panel
                header={
                  <span className="flex items-center gap-2 text-sm font-medium">
                    <PlayCircleOutlined className="text-blue-500" />
                    Test Your Guardrail
                  </span>
                }
                key="test"
              >
                <div className="space-y-3">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="block text-xs font-medium text-gray-600">Test Input (JSON)</label>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500">Load example:</span>
                        <button
                          type="button"
                          onClick={() => setTestInput(JSON.stringify(TEST_INPUT_EXAMPLES.pre_call.data, null, 2))}
                          className="px-2 py-1 text-xs rounded border border-orange-200 bg-orange-50 text-orange-700 hover:bg-orange-100 transition-colors"
                        >
                          Pre-call
                        </button>
                        <button
                          type="button"
                          onClick={() => setTestInput(JSON.stringify(TEST_INPUT_EXAMPLES.pre_mcp_call.data, null, 2))}
                          className="px-2 py-1 text-xs rounded border border-purple-200 bg-purple-50 text-purple-700 hover:bg-purple-100 transition-colors"
                        >
                          Pre MCP
                        </button>
                        <button
                          type="button"
                          onClick={() => setTestInput(JSON.stringify(TEST_INPUT_EXAMPLES.post_call.data, null, 2))}
                          className="px-2 py-1 text-xs rounded border border-green-200 bg-green-50 text-green-700 hover:bg-green-100 transition-colors"
                        >
                          Post-call
                        </button>
                      </div>
                    </div>
                    <div className="mb-2 p-2 bg-gray-50 rounded text-xs text-gray-600 border border-gray-200">
                      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                        <div><strong>texts</strong>: Message content (always)</div>
                        <div><strong>images</strong>: Base64 images (vision)</div>
                        <div><strong>tools</strong>: Tool definitions <span className="text-orange-600">(pre_call)</span>, MCP as OpenAI tool <span className="text-purple-600">(pre_mcp_call)</span></div>
                        <div><strong>tool_calls</strong>: LLM tool calls <span className="text-green-600">(post_call)</span></div>
                        <div><strong>structured_messages</strong>: Full messages <span className="text-orange-600">(pre_call)</span></div>
                        <div><strong>model</strong>: Model name (always)</div>
                      </div>
                    </div>
                    <TextArea
                      value={testInput}
                      onChange={(e) => setTestInput(e.target.value)}
                      rows={8}
                      className="font-mono text-xs"
                      placeholder='{"texts": ["test message"], ...}'
                    />
                  </div>
                  <div className="flex items-center gap-3">
                    <Button
                      size="xs"
                      onClick={handleTest}
                      disabled={isTesting}
                      icon={PlayCircleOutlined}
                    >
                      {isTesting ? "Running..." : "Run Test"}
                    </Button>
                    {testResult && (
                      <div className={`flex items-center gap-2 text-sm ${
                        testResult.error ? "text-red-600" :
                        testResult.action === "allow" ? "text-green-600" :
                        testResult.action === "block" ? "text-orange-600" :
                        "text-blue-600"
                      }`}>
                        {testResult.error ? (
                          <>
                            <CloseCircleOutlined />
                            <span>
                              {testResult.error_type && <span className="font-medium">[{testResult.error_type}] </span>}
                              {testResult.error}
                            </span>
                          </>
                        ) : testResult.action === "allow" ? (
                          <><CheckCircleOutlined /> Allowed</>
                        ) : testResult.action === "block" ? (
                          <><CloseCircleOutlined /> Blocked: {testResult.reason}</>
                        ) : testResult.action === "modify" ? (
                          <>
                            <CheckCircleOutlined /> Modified
                            {testResult.texts && testResult.texts.length > 0 && (
                              <span className="text-xs text-gray-500 ml-1">
                                → {testResult.texts[0].substring(0, 50)}{testResult.texts[0].length > 50 ? "..." : ""}
                              </span>
                            )}
                          </>
                        ) : (
                          <><CheckCircleOutlined /> {testResult.action || "Unknown"}</>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </Panel>
            </Collapse>
            {/* Contribution CTA Banner */}
            <div className="mt-3 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg flex items-center justify-between flex-shrink-0">
              <div className="flex items-center gap-3">
                <div className="bg-blue-100 rounded-full p-2">
                  <UsergroupAddOutlined className="text-blue-600 text-lg" />
                </div>
                <div>
                  <div className="text-sm font-medium text-gray-900">Built a useful guardrail?</div>
                  <div className="text-xs text-gray-600">Share it with the community and help others build faster</div>
                </div>
              </div>
              <Button
                size="xs"
                onClick={() => window.open('https://github.com/BerriAI/litellm-guardrails', '_blank')}
                icon={ExportOutlined}
                className="bg-blue-600 hover:bg-blue-700 text-white border-0"
              >
                Contribute Template
              </Button>
            </div>

          </div>

          {/* Primitives Panel */}
          <div className="w-[300px] flex-shrink-0 overflow-auto border-l border-gray-200 pl-6">
            <div className="flex items-center gap-2 mb-3">
              <CodeOutlined className="text-blue-500" />
              <span className="font-semibold text-gray-700">Available Primitives</span>
            </div>
            <p className="text-xs text-gray-500 mb-3">Click to copy functions to clipboard</p>
            
            <Collapse
              defaultActiveKey={["Return Values"]}
              className="primitives-collapse bg-transparent border-0"
              expandIconPosition="end"
            >
              {Object.entries(PRIMITIVES).map(([category, primitives]) => (
                <Panel
                  header={<span className="text-sm font-medium text-gray-700">{category}</span>}
                  key={category}
                  className="bg-white mb-2 rounded-lg border border-gray-200"
                >
                  <div className="space-y-2">
                    {primitives.map((p) => (
                      <button
                        key={p.name}
                        onClick={() => copyPrimitive(p.name)}
                        className={`w-full text-left px-2 py-2 rounded transition-colors ${
                          copiedPrimitive === p.name
                            ? "bg-green-100"
                            : "bg-gray-50 hover:bg-blue-50"
                        }`}
                      >
                        {copiedPrimitive === p.name ? (
                          <span className="flex items-center gap-1 text-xs font-mono text-green-700">
                            <CheckCircleOutlined /> Copied!
                          </span>
                        ) : (
                          <>
                            <div className="text-xs font-mono text-gray-800">{p.name}</div>
                            <div className="text-[10px] text-gray-500 mt-0.5">{p.desc}</div>
                          </>
                        )}
                      </button>
                    ))}
                  </div>
                </Panel>
              ))}
            </Collapse>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between pt-4 mt-4 border-t border-gray-200">
          <span className="text-xs text-gray-400">Changes are auto-saved to local draft</span>
          <div className="flex items-center gap-3">
            <Button variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button
              onClick={handleSave}
              loading={isSaving}
              disabled={isSaving || !guardrailName.trim()}
              icon={SaveOutlined}
            >
              {isEditMode ? "Update Guardrail" : "Save Guardrail"}
            </Button>
          </div>
        </div>
      </div>

      <style>{`
        .custom-code-modal .ant-modal-content {
          padding: 24px;
        }
        .custom-code-modal .ant-modal-close {
          top: 20px;
          right: 20px;
        }
        .primitives-collapse .ant-collapse-item {
          border: none !important;
        }
        .primitives-collapse .ant-collapse-header {
          padding: 8px 12px !important;
        }
        .primitives-collapse .ant-collapse-content-box {
          padding: 8px 12px !important;
        }
      `}</style>
    </Modal>
  );
};

export default CustomCodeModal;
