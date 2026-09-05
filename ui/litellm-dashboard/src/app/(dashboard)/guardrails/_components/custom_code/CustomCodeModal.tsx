import React, { useState, useRef, useEffect } from "react";
import { CheckCircle2, ChevronRight, Code, ExternalLink, PlayCircle, Save, Users, XCircle } from "lucide-react";
import { createGuardrailCall, updateGuardrailCall, testCustomCodeGuardrail } from "@/components/networking";
import { toast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  useComboboxAnchor,
} from "@/components/ui/combobox";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";

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

const TEMPLATE_ITEMS = Object.entries(CODE_TEMPLATES).map(([key, template]) => ({
  value: key,
  label: template.name,
}));

type ModeOption = (typeof MODE_OPTIONS)[number];

const MODE_OPTION_BY_VALUE: Record<string, ModeOption> = Object.fromEntries(
  MODE_OPTIONS.map((option) => [option.value, option]),
);

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

const CustomCodeModal: React.FC<CustomCodeModalProps> = ({ visible, onClose, onSuccess, accessToken, editData }) => {
  const anchor = useComboboxAnchor();
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
        texts: ['Tool: read_wiki_structure\nArguments: {"repoName": "BerriAI/litellm"}'],
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
                  repoName: { type: "string", description: "Repository name, e.g. BerriAI/litellm" },
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
          { role: "user", content: 'Tool: read_wiki_structure\nArguments: {"repoName": "BerriAI/litellm"}' },
        ],
        model: "mcp-tool-call",
      },
    },
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
      toast.fromError("Please enter a guardrail name");
      return;
    }
    if (!code.trim()) {
      toast.fromError("Please enter custom code");
      return;
    }
    if (!accessToken) {
      toast.fromError("No access token available");
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
        const modeChanged = mode.length !== existingMode.length || mode.some((m, i) => m !== existingMode[i]);
        if (modeChanged) {
          updateData.litellm_params.mode = mode;
        }
        if (defaultOn !== editData.litellm_params?.default_on) {
          updateData.litellm_params.default_on = defaultOn;
        }

        await updateGuardrailCall(accessToken, editData.guardrail_id, updateData);
        toast.success("Custom code guardrail updated successfully");
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
        toast.success("Custom code guardrail created successfully");
      }
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to save guardrail:", error);
      toast.fromError(
        `Failed to ${isEditMode ? "update" : "create"} guardrail: ` +
          (error instanceof Error ? error.message : String(error)),
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
      const testInputType: "request" | "response" = mode.some((m) => requestModes.includes(m))
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
  const selectedModeOptions = mode.map((value) => MODE_OPTION_BY_VALUE[value]).filter(Boolean);

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1400px]">
        <DialogHeader>
          <DialogTitle className="text-xl font-semibold">
            {isEditMode ? "Edit Custom Guardrail" : "Create Custom Guardrail"}
          </DialogTitle>
          <DialogDescription>Define custom logic using Python-like syntax</DialogDescription>
        </DialogHeader>

        {/* Top Controls */}
        <div className="flex items-center gap-4 border-b border-border py-4">
          <div className="max-w-[200px] flex-1">
            <label className="mb-1 block text-xs font-medium text-muted-foreground">Guardrail Name</label>
            <Input
              value={guardrailName}
              onChange={(e) => setGuardrailName(e.target.value)}
              placeholder="e.g., block-pii-custom"
            />
          </div>
          <div className="w-[280px]">
            <label className="mb-1 block text-xs font-medium text-muted-foreground">Mode (can select multiple)</label>
            <Combobox
              items={MODE_OPTIONS}
              value={selectedModeOptions}
              onValueChange={(options: ModeOption[]) => setMode(options.map((option) => option.value))}
              multiple
            >
              <ComboboxChips render={<div ref={anchor} />} className="w-full">
                {selectedModeOptions.map((option) => (
                  <ComboboxChip key={option.value} aria-label={option.label}>
                    {option.label}
                  </ComboboxChip>
                ))}
                <ComboboxChipsInput placeholder={mode.length === 0 ? "Select modes" : undefined} />
              </ComboboxChips>
              <ComboboxContent anchor={anchor}>
                <ComboboxEmpty>No matching modes</ComboboxEmpty>
                <ComboboxList>
                  {(option: ModeOption) => (
                    <ComboboxItem key={option.value} value={option}>
                      {option.label}
                    </ComboboxItem>
                  )}
                </ComboboxList>
              </ComboboxContent>
            </Combobox>
          </div>
          <div className="w-[180px]">
            <label className="mb-1 block text-xs font-medium text-muted-foreground">Template</label>
            <Select
              items={TEMPLATE_ITEMS}
              value={selectedTemplate}
              onValueChange={(value: string | null) => value && handleTemplateChange(value)}
            >
              <SelectTrigger className="w-full" aria-label="Template">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectLabel>STANDARD</SelectLabel>
                  {TEMPLATE_ITEMS.map((template) => (
                    <SelectItem key={template.value} value={template.value}>
                      {template.label}
                    </SelectItem>
                  ))}
                </SelectGroup>
                <SelectSeparator />
                <button
                  type="button"
                  onClick={() => window.open("https://models.litellm.ai/guardrails", "_blank")}
                  className="flex w-full items-center gap-1 rounded-sm px-2 py-1.5 text-xs text-primary hover:bg-accent"
                >
                  <Users className="size-3.5" />
                  <span>Browse Community templates</span>
                  <ExternalLink className="size-2.5" />
                </button>
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center gap-2 pt-5">
            <span className="text-sm text-muted-foreground">Default On</span>
            <Switch checked={defaultOn} onCheckedChange={setDefaultOn} aria-label="Default On" />
          </div>
        </div>

        {/* Main Content */}
        <div className="mt-4 flex gap-6">
          {/* Code Editor */}
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="mb-2 flex shrink-0 items-center justify-between">
              <span className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">Python Logic</span>
              <span className="text-xs text-muted-foreground">Restricted environment (no imports)</span>
            </div>
            <div
              className="relative rounded-lg overflow-hidden border border-gray-700 bg-[#1e1e1e] shrink-0"
              style={{ minHeight: "300px", maxHeight: "400px" }}
            >
              {/* Line numbers */}
              <div
                className="absolute left-0 top-0 bottom-0 w-12 bg-[#1e1e1e] border-r border-gray-700 text-right pr-3 pt-3 select-none overflow-hidden"
                style={{
                  fontFamily: "'Fira Code', 'Monaco', 'Consolas', monospace",
                  fontSize: "14px",
                  lineHeight: "1.6",
                }}
              >
                {Array.from({ length: Math.max(lineCount, 20) }, (_, i) => (
                  <div key={i + 1} className="text-muted-foreground h-[22.4px]">
                    {i + 1}
                  </div>
                ))}
              </div>
              {/* Code textarea */}
              <textarea
                ref={textareaRef}
                value={code}
                onChange={(e) => setCode(e.target.value)}
                onKeyDown={handleKeyDown}
                spellCheck={false}
                className="w-full h-full pl-14 pr-4 pt-3 pb-3 resize-none focus:outline-hidden bg-transparent text-gray-200"
                style={{
                  fontFamily: "'Fira Code', 'Monaco', 'Consolas', monospace",
                  fontSize: "14px",
                  lineHeight: "1.6",
                  tabSize: 4,
                }}
              />
            </div>

            {/* Test Section */}
            <Collapsible
              open={testExpanded}
              onOpenChange={setTestExpanded}
              className="mt-3 shrink-0 rounded-lg border border-border"
            >
              <CollapsibleTrigger className="flex w-full items-center gap-2 p-3 text-sm font-medium">
                <ChevronRight className={`size-4 transition-transform ${testExpanded ? "rotate-90" : ""}`} />
                <PlayCircle className="size-4 text-muted-foreground" />
                Test Your Guardrail
              </CollapsibleTrigger>
              <CollapsibleContent className="p-3 pt-0">
                <div className="space-y-3">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="block text-xs font-medium text-muted-foreground">Test Input (JSON)</label>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">Load example:</span>
                        <button
                          type="button"
                          onClick={() => setTestInput(JSON.stringify(TEST_INPUT_EXAMPLES.pre_call.data, null, 2))}
                          className="px-2 py-1 text-xs rounded-sm border border-warning/20 bg-warning/10 text-warning hover:bg-warning/15 transition-colors"
                        >
                          Pre-call
                        </button>
                        <button
                          type="button"
                          onClick={() => setTestInput(JSON.stringify(TEST_INPUT_EXAMPLES.pre_mcp_call.data, null, 2))}
                          className="px-2 py-1 text-xs rounded-sm border border-purple-200 bg-purple-50 text-purple-700 hover:bg-purple-100 transition-colors dark:border-purple-800 dark:bg-purple-950 dark:text-purple-300 dark:hover:bg-purple-900"
                        >
                          Pre MCP
                        </button>
                        <button
                          type="button"
                          onClick={() => setTestInput(JSON.stringify(TEST_INPUT_EXAMPLES.post_call.data, null, 2))}
                          className="px-2 py-1 text-xs rounded-sm border border-success/20 bg-success/10 text-success hover:bg-success/15 transition-colors"
                        >
                          Post-call
                        </button>
                      </div>
                    </div>
                    <div className="mb-2 rounded-sm border border-border bg-muted/40 p-2 text-xs text-muted-foreground">
                      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                        <div>
                          <strong>texts</strong>: Message content (always)
                        </div>
                        <div>
                          <strong>images</strong>: Base64 images (vision)
                        </div>
                        <div>
                          <strong>tools</strong>: Tool definitions <span className="text-warning">(pre_call)</span>, MCP
                          as OpenAI tool <span className="text-purple-600">(pre_mcp_call)</span>
                        </div>
                        <div>
                          <strong>tool_calls</strong>: LLM tool calls <span className="text-success">(post_call)</span>
                        </div>
                        <div>
                          <strong>structured_messages</strong>: Full messages{" "}
                          <span className="text-warning">(pre_call)</span>
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
                      className="font-mono text-xs field-sizing-fixed"
                      placeholder='{"texts": ["test message"], ...}'
                    />
                  </div>
                  <div className="flex items-center gap-3">
                    <Button size="sm" onClick={handleTest} disabled={isTesting} aria-busy={isTesting}>
                      {isTesting ? <UiLoadingSpinner className="size-4" /> : <PlayCircle />}
                      {isTesting ? "Running..." : "Run Test"}
                    </Button>
                    {testResult && (
                      <div
                        className={`flex items-center gap-2 text-sm ${
                          testResult.error
                            ? "text-destructive"
                            : testResult.action === "allow"
                              ? "text-success"
                              : testResult.action === "block"
                                ? "text-warning"
                                : "text-info"
                        }`}
                      >
                        {testResult.error ? (
                          <>
                            <XCircle className="size-4" />
                            <span>
                              {testResult.error_type && <span className="font-medium">[{testResult.error_type}] </span>}
                              {testResult.error}
                            </span>
                          </>
                        ) : testResult.action === "allow" ? (
                          <>
                            <CheckCircle2 className="size-4" /> Allowed
                          </>
                        ) : testResult.action === "block" ? (
                          <>
                            <XCircle className="size-4" /> Blocked: {testResult.reason}
                          </>
                        ) : testResult.action === "modify" ? (
                          <>
                            <CheckCircle2 className="size-4" /> Modified
                            {testResult.texts && testResult.texts.length > 0 && (
                              <span className="ml-1 text-xs text-muted-foreground">
                                -&gt; {testResult.texts[0].substring(0, 50)}
                                {testResult.texts[0].length > 50 ? "..." : ""}
                              </span>
                            )}
                          </>
                        ) : (
                          <>
                            <CheckCircle2 className="size-4" /> {testResult.action || "Unknown"}
                          </>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </CollapsibleContent>
            </Collapsible>
            {/* Contribution CTA Banner */}
            <div className="mt-3 flex shrink-0 items-center justify-between rounded-lg border border-info/20 bg-linear-to-r from-blue-50 to-indigo-50 p-4 dark:from-blue-950 dark:to-indigo-950">
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-info/15 p-2">
                  <Users className="size-5 text-info" />
                </div>
                <div>
                  <div className="text-sm font-medium">Built a useful guardrail?</div>
                  <div className="text-xs text-muted-foreground">
                    Share it with the community and help others build faster
                  </div>
                </div>
              </div>
              <Button size="sm" onClick={() => window.open("https://github.com/BerriAI/litellm-guardrails", "_blank")}>
                <ExternalLink />
                Contribute Template
              </Button>
            </div>
          </div>

          {/* Primitives Panel */}
          <div className="w-[300px] shrink-0 overflow-auto border-l border-border pl-6">
            <div className="mb-3 flex items-center gap-2">
              <Code className="size-4 text-muted-foreground" />
              <span className="font-semibold">Available Primitives</span>
            </div>
            <p className="mb-3 text-xs text-muted-foreground">Click to copy functions to clipboard</p>

            <div className="space-y-2">
              {Object.entries(PRIMITIVES).map(([category, primitives]) => (
                <Collapsible
                  key={category}
                  defaultOpen={category === "Return Values"}
                  className="rounded-lg border border-border"
                >
                  <CollapsibleTrigger className="group flex w-full items-center justify-between px-3 py-2 text-sm font-medium">
                    {category}
                    <ChevronRight className="size-4 transition-transform group-data-panel-open:rotate-90" />
                  </CollapsibleTrigger>
                  <CollapsibleContent className="px-3 pb-3">
                    <div className="space-y-2">
                      {primitives.map((p) => (
                        <button
                          key={p.name}
                          onClick={() => copyPrimitive(p.name)}
                          className={`w-full rounded-sm px-2 py-2 text-left transition-colors ${
                            copiedPrimitive === p.name ? "bg-accent" : "bg-muted/40 hover:bg-accent"
                          }`}
                        >
                          {copiedPrimitive === p.name ? (
                            <span className="flex items-center gap-1 font-mono text-xs">
                              <CheckCircle2 className="size-3.5" /> Copied!
                            </span>
                          ) : (
                            <>
                              <div className="font-mono text-xs">{p.name}</div>
                              <div className="mt-0.5 text-[10px] text-muted-foreground">{p.desc}</div>
                            </>
                          )}
                        </button>
                      ))}
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-4 flex items-center justify-between border-t border-border pt-4">
          <span className="text-xs text-muted-foreground">Changes are auto-saved to local draft</span>
          <div className="flex items-center gap-3">
            <Button variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={isSaving || !guardrailName.trim()} aria-busy={isSaving}>
              {isSaving ? <UiLoadingSpinner className="size-4" /> : <Save />}
              {isEditMode ? "Update Guardrail" : "Save Guardrail"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default CustomCodeModal;
