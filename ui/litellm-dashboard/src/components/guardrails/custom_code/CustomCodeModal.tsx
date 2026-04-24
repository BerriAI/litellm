import React, { useEffect, useRef, useState } from "react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import {
  Check as CheckCircleOutlined,
  Code as CodeOutlined,
  ExternalLink as ExportOutlined,
  PlayCircle as PlayCircleOutlined,
  Save as SaveOutlined,
  UserRoundPlus as UsergroupAddOutlined,
  X as CloseCircleOutlined,
  X,
} from "lucide-react";
import {
  createGuardrailCall,
  testCustomCodeGuardrail,
  updateGuardrailCall,
} from "../../networking";
import NotificationsManager from "../../molecules/notifications_manager";

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

const PRIMITIVES = {
  "Return Values": [
    { name: "allow()", desc: "Let request/response through" },
    { name: "block(reason)", desc: "Reject with message" },
    {
      name: "modify(texts=[], images=[], tool_calls=[])",
      desc: "Transform content",
    },
  ],
  "HTTP Requests (async)": [
    {
      name: "await http_request(url, method, headers, body)",
      desc: "Make async HTTP request",
    },
    { name: "await http_get(url, headers)", desc: "Async GET request" },
    { name: "await http_post(url, body, headers)", desc: "Async POST request" },
  ],
  "Regex Functions": [
    {
      name: "regex_match(text, pattern)",
      desc: "Returns True if pattern found",
    },
    {
      name: "regex_replace(text, pattern, replacement)",
      desc: "Replace all matches",
    },
    {
      name: "regex_find_all(text, pattern)",
      desc: "Return list of matches",
    },
  ],
  "JSON Functions": [
    {
      name: "json_parse(text)",
      desc: "Parse JSON string, returns None on error",
    },
    { name: "json_stringify(obj)", desc: "Convert to JSON string" },
    {
      name: "json_schema_valid(obj, schema)",
      desc: "Validate against JSON schema",
    },
  ],
  "URL Functions": [
    { name: "extract_urls(text)", desc: "Extract all URLs from text" },
    { name: "is_valid_url(url)", desc: "Check if URL is valid" },
    {
      name: "all_urls_valid(text)",
      desc: "Check all URLs in text are valid",
    },
  ],
  "Code Detection": [
    { name: "detect_code(text)", desc: "Returns True if code detected" },
    {
      name: "detect_code_languages(text)",
      desc: "Returns list of detected languages",
    },
    {
      name: 'contains_code_language(text, ["sql"])',
      desc: "Check for specific languages",
    },
  ],
  "Text Utilities": [
    { name: "contains(text, substring)", desc: "Check if substring exists" },
    {
      name: "contains_any(text, [substr1, substr2])",
      desc: "Check if any substring exists",
    },
    { name: "word_count(text)", desc: "Count words" },
    { name: "char_count(text)", desc: "Count characters" },
    {
      name: "lower(text) / upper(text) / trim(text)",
      desc: "String transforms",
    },
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
  editData?: EditGuardrailData | null;
}

/**
 * Chip-style multi-select for Mode options. Mirrors the multi-select
 * pattern used elsewhere in the migrated UI.
 */
function ModeMultiSelect({
  value,
  onChange,
}: {
  value: string[];
  onChange: (next: string[]) => void;
}) {
  const remaining = MODE_OPTIONS.filter((o) => !value.includes(o.value));
  return (
    <div className="space-y-2">
      <Select
        value=""
        onValueChange={(v) => {
          if (v) onChange([...value, v]);
        }}
      >
        <SelectTrigger>
          <SelectValue placeholder="Select modes" />
        </SelectTrigger>
        <SelectContent>
          {remaining.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              All modes selected
            </div>
          ) : (
            remaining.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
      {value.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {value.map((v) => {
            const opt = MODE_OPTIONS.find((o) => o.value === v);
            return (
              <Badge
                key={v}
                variant="secondary"
                className="flex items-center gap-1"
              >
                {opt?.label ?? v}
                <button
                  type="button"
                  onClick={() => onChange(value.filter((s) => s !== v))}
                  className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                  aria-label={`Remove ${opt?.label ?? v}`}
                >
                  <X size={12} />
                </button>
              </Badge>
            );
          })}
        </div>
      )}
    </div>
  );
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
                  location: { type: "string", description: "City name" },
                },
                required: ["location"],
              },
            },
          },
        ],
        tool_calls: [],
        structured_messages: [
          { role: "system", content: "You are a helpful assistant." },
          { role: "user", content: "Hello, my SSN is 123-45-6789" },
        ],
        model: "gpt-4",
      },
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
              arguments: '{"location": "San Francisco"}',
            },
          },
        ],
        structured_messages: [],
        model: "gpt-4",
      },
    },
    pre_mcp_call: {
      name: "Pre MCP (MCP tool as OpenAI tool)",
      data: {
        texts: [
          'Tool: read_wiki_structure\nArguments: {"repoName": "BerriAI/litellm"}',
        ],
        images: [],
        tools: [
          {
            type: "function",
            function: {
              name: "read_wiki_structure",
              description:
                "Read the structure of a GitHub repository (MCP tool passed as OpenAI tool)",
              parameters: {
                type: "object",
                properties: {
                  repoName: {
                    type: "string",
                    description: "Repository name, e.g. BerriAI/litellm",
                  },
                },
                required: ["repoName"],
              },
            },
          },
        ],
        tool_calls: [
          {
            id: "call_mcp_001",
            type: "function",
            function: {
              name: "read_wiki_structure",
              arguments: '{"repoName": "BerriAI/litellm"}',
            },
          },
        ],
        structured_messages: [
          {
            role: "user",
            content:
              'Tool: read_wiki_structure\nArguments: {"repoName": "BerriAI/litellm"}',
          },
        ],
        model: "mcp-tool-call",
      },
    },
  };

  const [testInput, setTestInput] = useState(
    JSON.stringify(TEST_INPUT_EXAMPLES.pre_call.data, null, 2),
  );
  const [testResult, setTestResult] = useState<any>(null);
  const [copiedPrimitive, setCopiedPrimitive] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleTemplateChange = (templateKey: string) => {
    setSelectedTemplate(templateKey);
    setCode(
      CODE_TEMPLATES[templateKey as keyof typeof CODE_TEMPLATES].code,
    );
  };

  const normalizeMode = (m: string | string[] | undefined): string[] => {
    if (m === undefined || m === null) return ["pre_call"];
    if (Array.isArray(m)) return m.length ? m : ["pre_call"];
    return [m];
  };

  useEffect(() => {
    if (visible) {
      if (editData) {
        setGuardrailName(editData.guardrail_name || "");
        setMode(normalizeMode(editData.litellm_params?.mode));
        setDefaultOn(editData.litellm_params?.default_on || false);
        setCode(
          editData.litellm_params?.custom_code || CODE_TEMPLATES.empty.code,
        );
        setSelectedTemplate("");
      } else {
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

  const copyPrimitive = async (primitive: string) => {
    try {
      await navigator.clipboard.writeText(primitive);
      setCopiedPrimitive(primitive);
      setTimeout(() => setCopiedPrimitive(null), 2000);
    } catch (err) {
      console.error("Failed to copy:", err);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Tab") {
      e.preventDefault();
      const textarea = e.currentTarget;
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const newValue =
        code.substring(0, start) + "    " + code.substring(end);
      setCode(newValue);
      setTimeout(() => {
        textarea.selectionStart = textarea.selectionEnd = start + 4;
      }, 0);
    }
  };

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
        const updateData: any = {
          litellm_params: {
            custom_code: code,
          },
        };

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
        NotificationsManager.success(
          "Custom code guardrail updated successfully",
        );
      } else {
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
        NotificationsManager.success(
          "Custom code guardrail created successfully",
        );
      }
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to save guardrail:", error);
      NotificationsManager.fromBackend(
        `Failed to ${isEditMode ? "update" : "create"} guardrail: ` +
          (error instanceof Error ? error.message : String(error)),
      );
    } finally {
      setIsSaving(false);
    }
  };

  const handleTest = async () => {
    if (!accessToken) {
      setTestResult({ error: "No access token available" });
      return;
    }

    setIsTesting(true);
    setTestResult(null);

    try {
      let parsedInput;
      try {
        parsedInput = JSON.parse(testInput);
      } catch (e) {
        setTestResult({ error: "Invalid test input JSON" });
        setIsTesting(false);
        return;
      }

      if (!parsedInput.texts) {
        parsedInput.texts = [];
      }

      const requestModes = ["pre_call", "pre_mcp_call"];
      const responseModes = ["post_call", "post_mcp_call"];
      const testInputType: "request" | "response" = mode.some((m) =>
        requestModes.includes(m),
      )
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
        error:
          error instanceof Error ? error.message : "Failed to test custom code",
      });
    } finally {
      setIsTesting(false);
    }
  };

  const lineCount = code.split("\n").length;

  return (
    <Dialog
      open={visible}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="max-w-[1400px] w-[95vw]">
        <DialogHeader>
          <DialogTitle>
            {isEditMode ? "Edit Custom Guardrail" : "Create Custom Guardrail"}
          </DialogTitle>
          <p className="text-sm text-muted-foreground mt-1">
            Define custom logic using Python-like syntax
          </p>
        </DialogHeader>
        <div className="flex flex-col h-[80vh]">
          <div className="flex items-center gap-4 py-4 border-b border-border flex-wrap">
            <div className="flex-1 max-w-[200px] min-w-[180px]">
              <Label htmlFor="custom-code-name" className="text-xs mb-1">
                Guardrail Name
              </Label>
              <Input
                id="custom-code-name"
                value={guardrailName}
                onChange={(e) => setGuardrailName(e.target.value)}
                placeholder="e.g., block-pii-custom"
              />
            </div>
            <div className="w-[280px]">
              <Label className="text-xs mb-1">Mode (can select multiple)</Label>
              <ModeMultiSelect value={mode} onChange={setMode} />
            </div>
            <div className="w-[200px]">
              <Label className="text-xs mb-1">Template</Label>
              <Select
                value={selectedTemplate}
                onValueChange={handleTemplateChange}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select template" />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectLabel>STANDARD</SelectLabel>
                    {Object.entries(CODE_TEMPLATES).map(([key, template]) => (
                      <SelectItem key={key} value={key}>
                        {template.name}
                      </SelectItem>
                    ))}
                  </SelectGroup>
                  <Separator className="my-2" />
                  <button
                    type="button"
                    className="flex w-full items-center gap-1 px-3 py-2 text-xs text-primary hover:bg-muted"
                    onClick={(e) => {
                      e.preventDefault();
                      window.open(
                        "https://models.litellm.ai/guardrails",
                        "_blank",
                      );
                    }}
                  >
                    <UsergroupAddOutlined className="h-3 w-3" />
                    <span>Browse Community templates</span>
                    <ExportOutlined className="h-3 w-3" />
                  </button>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center gap-2 pt-5">
              <span className="text-sm text-muted-foreground">Default On</span>
              <Switch checked={defaultOn} onCheckedChange={setDefaultOn} />
            </div>
          </div>

          <div className="flex flex-1 overflow-hidden mt-4 gap-6">
            <div className="flex-[2] flex flex-col min-w-0 overflow-y-auto">
              <div className="flex items-center justify-between mb-2 flex-shrink-0">
                <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                  Python Logic
                </span>
                <span className="text-xs text-muted-foreground">
                  Restricted environment (no imports)
                </span>
              </div>
              {/* Categorical dark code-editor chrome — not a semantic token. */}
              <div
                className="relative rounded-lg overflow-hidden border border-border bg-[#1e1e1e] flex-shrink-0"
                style={{ minHeight: "300px", maxHeight: "400px" }}
              >
                <div
                  className="absolute left-0 top-0 bottom-0 w-12 bg-[#1e1e1e] border-r border-border text-right pr-3 pt-3 select-none overflow-hidden"
                  style={{
                    fontFamily:
                      "'Fira Code', 'Monaco', 'Consolas', monospace",
                    fontSize: "14px",
                    lineHeight: "1.6",
                  }}
                >
                  {Array.from({ length: Math.max(lineCount, 20) }, (_, i) => (
                    <div
                      key={i + 1}
                      className="text-muted-foreground h-[22.4px]"
                    >
                      {i + 1}
                    </div>
                  ))}
                </div>
                <textarea
                  ref={textareaRef}
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                  onKeyDown={handleKeyDown}
                  spellCheck={false}
                  className="w-full h-full pl-14 pr-4 pt-3 pb-3 resize-none focus:outline-none bg-transparent text-gray-200"
                  style={{
                    fontFamily:
                      "'Fira Code', 'Monaco', 'Consolas', monospace",
                    fontSize: "14px",
                    lineHeight: "1.6",
                    tabSize: 4,
                  }}
                />
              </div>

              <Accordion
                type="multiple"
                value={testExpanded ? ["test"] : []}
                onValueChange={(values) =>
                  setTestExpanded(values.includes("test"))
                }
                className="mt-3 bg-card border border-border rounded-lg flex-shrink-0 px-4"
              >
                <AccordionItem value="test" className="border-b-0">
                  <AccordionTrigger>
                    <span className="flex items-center gap-2 text-sm font-medium">
                      <PlayCircleOutlined className="text-primary h-4 w-4" />
                      Test Your Guardrail
                    </span>
                  </AccordionTrigger>
                  <AccordionContent>
                    <div className="space-y-3">
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <Label className="text-xs">Test Input (JSON)</Label>
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-muted-foreground">
                              Load example:
                            </span>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="h-6 px-2 text-xs"
                              onClick={() =>
                                setTestInput(
                                  JSON.stringify(
                                    TEST_INPUT_EXAMPLES.pre_call.data,
                                    null,
                                    2,
                                  ),
                                )
                              }
                            >
                              Pre-call
                            </Button>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="h-6 px-2 text-xs"
                              onClick={() =>
                                setTestInput(
                                  JSON.stringify(
                                    TEST_INPUT_EXAMPLES.pre_mcp_call.data,
                                    null,
                                    2,
                                  ),
                                )
                              }
                            >
                              Pre MCP
                            </Button>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="h-6 px-2 text-xs"
                              onClick={() =>
                                setTestInput(
                                  JSON.stringify(
                                    TEST_INPUT_EXAMPLES.post_call.data,
                                    null,
                                    2,
                                  ),
                                )
                              }
                            >
                              Post-call
                            </Button>
                          </div>
                        </div>
                        <div className="mb-2 p-2 bg-muted rounded text-xs text-muted-foreground border border-border">
                          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                            <div>
                              <strong>texts</strong>: Message content (always)
                            </div>
                            <div>
                              <strong>images</strong>: Base64 images (vision)
                            </div>
                            <div>
                              <strong>tools</strong>: Tool definitions{" "}
                              <span>(pre_call)</span>, MCP as OpenAI tool{" "}
                              <span>(pre_mcp_call)</span>
                            </div>
                            <div>
                              <strong>tool_calls</strong>: LLM tool calls{" "}
                              <span>(post_call)</span>
                            </div>
                            <div>
                              <strong>structured_messages</strong>: Full
                              messages <span>(pre_call)</span>
                            </div>
                            <div>
                              <strong>model</strong>: Model name (always)
                            </div>
                          </div>
                        </div>
                        <Textarea
                          value={testInput}
                          onChange={(e) => setTestInput(e.target.value)}
                          rows={8}
                          className="font-mono text-xs"
                          placeholder='{"texts": ["test message"], ...}'
                        />
                      </div>
                      <div className="flex items-center gap-3">
                        <Button
                          size="sm"
                          onClick={handleTest}
                          disabled={isTesting}
                        >
                          <PlayCircleOutlined className="h-4 w-4 mr-1" />
                          {isTesting ? "Running..." : "Run Test"}
                        </Button>
                        {testResult && (
                          <TestResultDisplay result={testResult} />
                        )}
                      </div>
                    </div>
                  </AccordionContent>
                </AccordionItem>
              </Accordion>

              <div className="mt-3 p-4 bg-muted border border-border rounded-lg flex items-center justify-between flex-shrink-0">
                <div className="flex items-center gap-3">
                  <div className="bg-primary/10 rounded-full p-2">
                    <UsergroupAddOutlined className="text-primary h-5 w-5" />
                  </div>
                  <div>
                    <div className="text-sm font-medium text-foreground">
                      Built a useful guardrail?
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Share it with the community and help others build faster
                    </div>
                  </div>
                </div>
                <Button
                  size="sm"
                  onClick={() =>
                    window.open(
                      "https://github.com/BerriAI/litellm-guardrails",
                      "_blank",
                    )
                  }
                >
                  <ExportOutlined className="h-4 w-4 mr-1" />
                  Contribute Template
                </Button>
              </div>
            </div>

            <div className="w-[300px] flex-shrink-0 overflow-auto border-l border-border pl-6">
              <div className="flex items-center gap-2 mb-3">
                <CodeOutlined className="text-primary h-4 w-4" />
                <span className="font-semibold text-foreground">
                  Available Primitives
                </span>
              </div>
              <p className="text-xs text-muted-foreground mb-3">
                Click to copy functions to clipboard
              </p>

              <Accordion
                type="multiple"
                defaultValue={["Return Values"]}
                className="bg-transparent border-0"
              >
                {Object.entries(PRIMITIVES).map(([category, primitives]) => (
                  <AccordionItem
                    key={category}
                    value={category}
                    className="bg-card mb-2 rounded-lg border border-border px-3"
                  >
                    <AccordionTrigger className="py-2">
                      <span className="text-sm font-medium text-foreground">
                        {category}
                      </span>
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="space-y-2">
                        {primitives.map((p) => (
                          <button
                            key={p.name}
                            onClick={() => copyPrimitive(p.name)}
                            className={
                              copiedPrimitive === p.name
                                ? "w-full text-left px-2 py-2 rounded transition-colors bg-primary/10"
                                : "w-full text-left px-2 py-2 rounded transition-colors bg-muted hover:bg-primary/10"
                            }
                          >
                            {copiedPrimitive === p.name ? (
                              <span className="flex items-center gap-1 text-xs font-mono text-primary">
                                <CheckCircleOutlined className="h-3 w-3" />{" "}
                                Copied!
                              </span>
                            ) : (
                              <>
                                <div className="text-xs font-mono text-foreground">
                                  {p.name}
                                </div>
                                <div className="text-[10px] text-muted-foreground mt-0.5">
                                  {p.desc}
                                </div>
                              </>
                            )}
                          </button>
                        ))}
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                ))}
              </Accordion>
            </div>
          </div>

          <div className="flex items-center justify-between pt-4 mt-4 border-t border-border">
            <span className="text-xs text-muted-foreground">
              Changes are auto-saved to local draft
            </span>
            <div className="flex items-center gap-3">
              <Button variant="outline" onClick={onClose}>
                Cancel
              </Button>
              <Button
                onClick={handleSave}
                disabled={isSaving || !guardrailName.trim()}
              >
                <SaveOutlined className="h-4 w-4 mr-1" />
                {isEditMode ? "Update Guardrail" : "Save Guardrail"}
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

function TestResultDisplay({ result }: { result: any }) {
  const base = "flex items-center gap-2 text-sm";
  if (result.error) {
    return (
      <div className={`${base} text-destructive`}>
        <CloseCircleOutlined className="h-4 w-4" />
        <span>
          {result.error_type && (
            <span className="font-medium">[{result.error_type}] </span>
          )}
          {result.error}
        </span>
      </div>
    );
  }
  if (result.action === "allow") {
    return (
      <div className={`${base} text-green-600 dark:text-green-400`}>
        <CheckCircleOutlined className="h-4 w-4" /> Allowed
      </div>
    );
  }
  if (result.action === "block") {
    return (
      <div className={`${base} text-orange-600 dark:text-orange-400`}>
        <CloseCircleOutlined className="h-4 w-4" /> Blocked: {result.reason}
      </div>
    );
  }
  if (result.action === "modify") {
    return (
      <div className={`${base} text-primary`}>
        <CheckCircleOutlined className="h-4 w-4" /> Modified
        {result.texts && result.texts.length > 0 && (
          <span className="text-xs text-muted-foreground ml-1">
            → {result.texts[0].substring(0, 50)}
            {result.texts[0].length > 50 ? "..." : ""}
          </span>
        )}
      </div>
    );
  }
  return (
    <div className={`${base} text-primary`}>
      <CheckCircleOutlined className="h-4 w-4" /> {result.action || "Unknown"}
    </div>
  );
}

export default CustomCodeModal;
