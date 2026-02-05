import React, { useState, useRef, useEffect } from "react";
import { Modal, Select, Switch, Collapse, Input, Spin } from "antd";
import { Button, TextInput } from "@tremor/react";
import {
  CodeOutlined,
  PlayCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  CaretRightOutlined,
  SaveOutlined,
} from "@ant-design/icons";
import { createGuardrailCall, testCustomCodeGuardrail } from "../../networking";
import NotificationsManager from "../../molecules/notifications_manager";

const { Panel } = Collapse;
const { TextArea } = Input;

// Code templates
const CODE_TEMPLATES = {
  empty: {
    name: "Empty Template",
    code: `def apply_guardrail(inputs, request_data, input_type):
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
};

// Available primitives organized by category
const PRIMITIVES = {
  "Return Values": [
    { name: "allow()", desc: "Let request/response through" },
    { name: "block(reason)", desc: "Reject with message" },
    { name: "modify(texts=[], images=[], tool_calls=[])", desc: "Transform content" },
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
];

interface CustomCodeModalProps {
  visible: boolean;
  onClose: () => void;
  onSuccess: () => void;
  accessToken: string | null;
}

const CustomCodeModal: React.FC<CustomCodeModalProps> = ({
  visible,
  onClose,
  onSuccess,
  accessToken,
}) => {
  const [guardrailName, setGuardrailName] = useState("");
  const [mode, setMode] = useState<string>("pre_call");
  const [defaultOn, setDefaultOn] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<string>("empty");
  const [code, setCode] = useState(CODE_TEMPLATES.empty.code);
  const [isSaving, setIsSaving] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [testExpanded, setTestExpanded] = useState(false);
  const [testInput, setTestInput] = useState('{"texts": ["Hello, my SSN is 123-45-6789"], "images": [], "tools": [], "tool_calls": [], "structured_messages": [], "model": "gpt-4"}');
  const [testResult, setTestResult] = useState<any>(null);
  const [copiedPrimitive, setCopiedPrimitive] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Handle template change
  const handleTemplateChange = (templateKey: string) => {
    setSelectedTemplate(templateKey);
    setCode(CODE_TEMPLATES[templateKey as keyof typeof CODE_TEMPLATES].code);
  };

  // Reset form when modal opens
  useEffect(() => {
    if (visible) {
      setGuardrailName("");
      setMode("pre_call");
      setDefaultOn(false);
      setSelectedTemplate("empty");
      setCode(CODE_TEMPLATES.empty.code);
      setTestResult(null);
      setTestExpanded(false);
    }
  }, [visible]);

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

  // Save guardrail
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
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to create guardrail:", error);
      NotificationsManager.fromBackend(
        "Failed to create guardrail: " + (error instanceof Error ? error.message : String(error))
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

      const response = await testCustomCodeGuardrail(accessToken, {
        custom_code: code,
        test_input: parsedInput,
        input_type: mode as "request" | "response",
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
      width={1200}
      className="custom-code-modal"
      closable={true}
      destroyOnClose
    >
      <div className="flex flex-col h-[80vh]">
        {/* Header */}
        <div className="pb-4 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900">Create Custom Guardrail</h2>
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
          <div className="w-[180px]">
            <label className="block text-xs font-medium text-gray-600 mb-1">Mode</label>
            <Select
              value={mode}
              onChange={setMode}
              options={MODE_OPTIONS}
              className="w-full"
              size="middle"
            />
          </div>
          <div className="w-[180px]">
            <label className="block text-xs font-medium text-gray-600 mb-1">Template</label>
            <Select
              value={selectedTemplate}
              onChange={handleTemplateChange}
              className="w-full"
              size="middle"
            >
              {Object.entries(CODE_TEMPLATES).map(([key, template]) => (
                <Select.Option key={key} value={key}>
                  {template.name}
                </Select.Option>
              ))}
            </Select>
          </div>
          <div className="flex items-center gap-2 pt-5">
            <span className="text-sm text-gray-600">Default On</span>
            <Switch checked={defaultOn} onChange={setDefaultOn} />
          </div>
        </div>

        {/* Main Content */}
        <div className="flex flex-1 overflow-hidden mt-4 gap-4">
          {/* Code Editor */}
          <div className="flex-1 flex flex-col min-w-0">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Python Logic</span>
              <span className="text-xs text-gray-400">Restricted environment (no imports)</span>
            </div>
            <div className="flex-1 relative rounded-lg overflow-hidden border border-gray-700 bg-[#1e1e1e]">
              {/* Line numbers */}
              <div 
                className="absolute left-0 top-0 bottom-0 w-10 bg-[#1e1e1e] border-r border-gray-700 text-right pr-2 pt-3 select-none overflow-hidden"
                style={{ fontFamily: "monospace", fontSize: "13px", lineHeight: "1.5" }}
              >
                {Array.from({ length: Math.max(lineCount, 20) }, (_, i) => (
                  <div key={i + 1} className="text-gray-500 h-[19.5px]">{i + 1}</div>
                ))}
              </div>
              {/* Code textarea */}
              <textarea
                ref={textareaRef}
                value={code}
                onChange={(e) => setCode(e.target.value)}
                onKeyDown={handleKeyDown}
                spellCheck={false}
                className="w-full h-full pl-12 pr-4 pt-3 pb-3 font-mono text-sm resize-none focus:outline-none bg-transparent text-gray-200"
                style={{ lineHeight: "1.5", tabSize: 4 }}
              />
            </div>

            {/* Test Section */}
            <Collapse
              activeKey={testExpanded ? ["test"] : []}
              onChange={(keys) => setTestExpanded(keys.includes("test"))}
              className="mt-3 bg-white border border-gray-200 rounded-lg"
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
                    <label className="block text-xs font-medium text-gray-600 mb-1">Test Input (JSON)</label>
                    <TextArea
                      value={testInput}
                      onChange={(e) => setTestInput(e.target.value)}
                      rows={4}
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
          </div>

          {/* Primitives Panel */}
          <div className="w-[280px] flex-shrink-0 overflow-auto">
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
              Save Guardrail
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
