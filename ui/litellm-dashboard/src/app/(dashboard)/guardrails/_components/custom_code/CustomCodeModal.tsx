import React, { useState, useRef, useEffect, useMemo } from "react";
import { Trans, useTranslation } from "react-i18next";
import { CheckCircle2, ChevronRight, Code, ExternalLink, PlayCircle, Save, Users, XCircle } from "lucide-react";
import { createGuardrailCall, updateGuardrailCall, testCustomCodeGuardrail } from "@/components/networking";
import { toast } from "@/lib/toast";
import { loggingOnlyContinueFromParams, loggingOnlyScopeToChoice } from "../guardrail_info_helpers";
import type { LoggingOnlyScope, LoggingOnlyScopeChoice } from "../guardrail_info_helpers";
import {
  CustomCodeLoggingOnlyScopeSelect,
  getCustomCodeLoggingOnlyScopeCreate,
  getCustomCodeLoggingOnlyScopeUpdate,
} from "./CustomCodeLoggingOnlyScope";
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
import { StreamScopeFields } from "../StreamScopeFields";
import {
  formatGuardrailMode,
  streamScopeByModeFromConfig,
  streamScopeForUpdate,
  streamScopePayload,
  type GuardrailStreamScope,
} from "../guardrail_info_helpers";
import {
  CODE_TEMPLATES,
  DEFAULT_OPEN_PRIMITIVE_CATEGORY,
  MODE_OPTIONS,
  PRIMITIVE_CATEGORIES,
  TEMPLATE_ITEMS,
  type ModeOption,
} from "./custom_code_catalog";

// Data for editing an existing guardrail

export interface EditGuardrailData {
  guardrail_id: string;
  guardrail_name: string;
  litellm_params: {
    mode?: string | string[] | Record<string, unknown>;
    default_on?: boolean;
    custom_code?: string;
    logging_only_scope?: LoggingOnlyScope | null;
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
  const { t } = useTranslation();
  const anchor = useComboboxAnchor();
  const isEditMode = !!editData;
  const [guardrailName, setGuardrailName] = useState("");
  const [mode, setMode] = useState<string[]>(["pre_call"]);
  const [streamScopeByMode, setStreamScopeByMode] = useState<Record<string, GuardrailStreamScope>>({});
  const [loggingOnlyScopeChoice, setLoggingOnlyScopeChoice] = useState<LoggingOnlyScopeChoice>("default");
  const [loggingOnlyContinue, setLoggingOnlyContinue] = useState(false);
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

  // Normalize mode from API (string or string[]) to string[].
  // A tag-scoped mode dict ({ tags, default }) is managed outside this editor, so it
  // contributes no editable modes and is displayed read-only instead.
  const normalizeMode = (m: string | string[] | Record<string, unknown> | undefined): string[] => {
    if (m === undefined || m === null) return ["pre_call"];
    if (Array.isArray(m)) return m.length ? m : ["pre_call"];
    if (typeof m === "string") return [m];
    return [];
  };

  // Reset form when modal opens or editData changes
  useEffect(() => {
    if (visible) {
      if (editData) {
        // Edit mode: populate with existing data
        setGuardrailName(editData.guardrail_name || "");
        setMode(normalizeMode(editData.litellm_params?.mode));
        setStreamScopeByMode(
          streamScopeByModeFromConfig(
            editData.litellm_params?.stream_scope,
            normalizeMode(editData.litellm_params?.mode),
          ),
        );
        setLoggingOnlyScopeChoice(loggingOnlyScopeToChoice(editData.litellm_params?.logging_only_scope));
        setLoggingOnlyContinue(loggingOnlyContinueFromParams(editData.litellm_params));
        setDefaultOn(editData.litellm_params?.default_on || false);
        setCode(editData.litellm_params?.custom_code || CODE_TEMPLATES.empty.code);
        setSelectedTemplate(""); // No template selected in edit mode
      } else {
        // Create mode: reset to defaults
        setGuardrailName("");
        setMode(["pre_call"]);
        setStreamScopeByMode({});
        setLoggingOnlyScopeChoice("default");
        setLoggingOnlyContinue(false);
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
      toast.fromError(t("guardrails.customCode.toast.nameRequired"));
      return;
    }
    if (!code.trim()) {
      toast.fromError(t("guardrails.customCode.toast.codeRequired"));
      return;
    }
    if (!accessToken) {
      toast.fromError(t("guardrails.customCode.toast.noAccessToken"));
      return;
    }

    setIsSaving(true);
    try {
      if (isEditMode && editData) {
        // Update existing guardrail
        const updateData: any = {
          litellm_params: {
            custom_code: code,
            ...getCustomCodeLoggingOnlyScopeUpdate(
              mode,
              editData.litellm_params,
              loggingOnlyScopeChoice,
              loggingOnlyContinue,
            ),
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
        const nextStreamScope = streamScopeForUpdate(
          mode,
          streamScopeByMode,
          editData.litellm_params?.stream_scope,
          existingMode,
        );
        if (nextStreamScope !== undefined) {
          updateData.litellm_params.stream_scope = nextStreamScope;
        }

        await updateGuardrailCall(accessToken, editData.guardrail_id, updateData);
        toast.success(t("guardrails.customCode.toast.updated"));
      } else {
        // Create new guardrail
        const streamScope = streamScopePayload(mode, streamScopeByMode);
        const guardrailData = {
          guardrail_name: guardrailName,
          litellm_params: {
            guardrail: "custom_code",
            mode: mode,
            default_on: defaultOn,
            custom_code: code,
            ...(streamScope !== undefined ? { stream_scope: streamScope } : {}),
            ...getCustomCodeLoggingOnlyScopeCreate(mode, loggingOnlyScopeChoice, loggingOnlyContinue),
          },
          guardrail_info: {},
        };

        await createGuardrailCall(accessToken, guardrailData);
        toast.success(t("guardrails.customCode.toast.created"));
      }
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Failed to save guardrail:", error);
      toast.fromError(
        t(isEditMode ? "guardrails.customCode.toast.updateFailed" : "guardrails.customCode.toast.createFailed", {
          error: error instanceof Error ? error.message : String(error),
        }),
      );
    } finally {
      setIsSaving(false);
    }
  };

  // Test guardrail using backend endpoint
  const handleTest = async () => {
    if (!accessToken) {
      setTestResult({ error: t("guardrails.customCode.test.errors.noAccessToken") });
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
        setTestResult({ error: t("guardrails.customCode.test.errors.invalidInput") });
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
        setTestResult({ error: t("guardrails.customCode.test.errors.unknown") });
      }
    } catch (error) {
      console.error("Failed to test custom code:", error);
      setTestResult({
        error: error instanceof Error ? error.message : t("guardrails.customCode.test.errors.failed"),
      });
    } finally {
      setIsTesting(false);
    }
  };

  const lineCount = code.split("\n").length;
  const modeOptions: ModeOption[] = useMemo(
    () => MODE_OPTIONS.map(({ value, labelKey }) => ({ value, label: t(labelKey) })),
    [t],
  );
  const modeOptionByValue: Record<string, ModeOption> = useMemo(
    () => Object.fromEntries(modeOptions.map((option) => [option.value, option])),
    [modeOptions],
  );
  const templateItems = useMemo(
    () => TEMPLATE_ITEMS.map(({ value, labelKey }) => ({ value, label: t(labelKey) })),
    [t],
  );
  const selectedModeOptions = mode.map((value) => modeOptionByValue[value]).filter(Boolean);
  const rawEditMode = editData?.litellm_params?.mode;
  const tagScopedModeLabel =
    rawEditMode !== null && typeof rawEditMode === "object" && !Array.isArray(rawEditMode)
      ? formatGuardrailMode(rawEditMode) || "-"
      : null;

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1400px]">
        <DialogHeader>
          <DialogTitle className="text-xl font-semibold">
            {isEditMode ? t("guardrails.customCode.title.edit") : t("guardrails.customCode.title.create")}
          </DialogTitle>
          <DialogDescription>{t("guardrails.customCode.description")}</DialogDescription>
        </DialogHeader>

        {/* Top Controls */}
        <div className="flex flex-wrap items-center gap-4 border-b border-border py-4">
          <div className="min-w-[160px] max-w-[200px] flex-1">
            <label className="mb-1 block text-xs font-medium text-muted-foreground">
              {t("guardrails.customCode.nameLabel")}
            </label>
            <Input
              value={guardrailName}
              onChange={(e) => setGuardrailName(e.target.value)}
              placeholder={t("guardrails.customCode.namePlaceholder")}
            />
          </div>
          <div className="w-[280px]">
            <label className="mb-1 block text-xs font-medium text-muted-foreground">
              {tagScopedModeLabel
                ? t("guardrails.customCode.modeLabel.tagScoped")
                : t("guardrails.customCode.modeLabel.multiple")}
            </label>
            {tagScopedModeLabel ? (
              <Input
                value={tagScopedModeLabel}
                disabled
                aria-label={t("guardrails.customCode.tagScopedModeAriaLabel")}
              />
            ) : (
              <Combobox
                items={modeOptions}
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
                  <ComboboxChipsInput
                    placeholder={mode.length === 0 ? t("guardrails.customCode.modePlaceholder") : undefined}
                  />
                </ComboboxChips>
                <ComboboxContent anchor={anchor}>
                  <ComboboxEmpty>{t("guardrails.customCode.noMatchingModes")}</ComboboxEmpty>
                  <ComboboxList>
                    {(option: ModeOption) => (
                      <ComboboxItem key={option.value} value={option}>
                        {option.label}
                      </ComboboxItem>
                    )}
                  </ComboboxList>
                </ComboboxContent>
              </Combobox>
            )}
          </div>
          {mode.includes("logging_only") && (
            <CustomCodeLoggingOnlyScopeSelect value={loggingOnlyScopeChoice} onChange={setLoggingOnlyScopeChoice} />
          )}
          {mode.includes("logging_only") && (
            <div className="flex shrink-0 items-center gap-2 pt-5">
              <span className="text-sm text-muted-foreground">{t("guardrails.customCode.continueObserving")}</span>
              <Switch
                checked={loggingOnlyScopeChoice === "default" && loggingOnlyContinue}
                disabled={loggingOnlyScopeChoice !== "default"}
                onCheckedChange={setLoggingOnlyContinue}
                aria-label={t("guardrails.customCode.continueObserving")}
              />
            </div>
          )}
          <div className="w-[180px]">
            <label className="mb-1 block text-xs font-medium text-muted-foreground">
              {t("guardrails.customCode.templateLabel")}
            </label>
            <Select
              items={templateItems}
              value={selectedTemplate}
              onValueChange={(value: string | null) => value && handleTemplateChange(value)}
            >
              <SelectTrigger className="w-full" aria-label={t("guardrails.customCode.templateLabel")}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  <SelectLabel>{t("guardrails.customCode.standardTemplates")}</SelectLabel>
                  {templateItems.map((template) => (
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
                  <span>{t("guardrails.customCode.browseCommunity")}</span>
                  <ExternalLink className="size-2.5" />
                </button>
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center gap-2 pt-5">
            <span className="text-sm text-muted-foreground">{t("guardrails.customCode.defaultOn")}</span>
            <Switch
              checked={defaultOn}
              onCheckedChange={setDefaultOn}
              aria-label={t("guardrails.customCode.defaultOn")}
            />
          </div>
        </div>
        {mode.length > 0 && (
          <div className="border-b border-border py-4">
            <StreamScopeFields modes={mode} value={streamScopeByMode} onChange={setStreamScopeByMode} />
          </div>
        )}

        {/* Main Content */}
        <div className="mt-4 flex gap-6">
          {/* Code Editor */}
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="mb-2 flex shrink-0 items-center justify-between">
              <span className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
                {t("guardrails.customCode.pythonLogic")}
              </span>
              <span className="text-xs text-muted-foreground">{t("guardrails.customCode.restrictedEnvironment")}</span>
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
                {t("guardrails.customCode.test.title")}
              </CollapsibleTrigger>
              <CollapsibleContent className="p-3 pt-0">
                <div className="space-y-3">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label className="block text-xs font-medium text-muted-foreground">
                        {t("guardrails.customCode.test.inputLabel")}
                      </label>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted-foreground">
                          {t("guardrails.customCode.test.loadExample")}
                        </span>
                        <button
                          type="button"
                          onClick={() => setTestInput(JSON.stringify(TEST_INPUT_EXAMPLES.pre_call.data, null, 2))}
                          className="px-2 py-1 text-xs rounded-sm border border-warning/20 bg-warning/10 text-warning hover:bg-warning/15 transition-colors"
                        >
                          {t("guardrails.customCode.test.examples.preCall")}
                        </button>
                        <button
                          type="button"
                          onClick={() => setTestInput(JSON.stringify(TEST_INPUT_EXAMPLES.pre_mcp_call.data, null, 2))}
                          className="px-2 py-1 text-xs rounded-sm border border-purple-200 bg-purple-50 text-purple-700 hover:bg-purple-100 transition-colors dark:border-purple-800 dark:bg-purple-950 dark:text-purple-300 dark:hover:bg-purple-900"
                        >
                          {t("guardrails.customCode.test.examples.preMcp")}
                        </button>
                        <button
                          type="button"
                          onClick={() => setTestInput(JSON.stringify(TEST_INPUT_EXAMPLES.post_call.data, null, 2))}
                          className="px-2 py-1 text-xs rounded-sm border border-success/20 bg-success/10 text-success hover:bg-success/15 transition-colors"
                        >
                          {t("guardrails.customCode.test.examples.postCall")}
                        </button>
                      </div>
                    </div>
                    <div className="mb-2 rounded-sm border border-border bg-muted/40 p-2 text-xs text-muted-foreground">
                      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                        <div>
                          <Trans
                            i18nKey="guardrails.customCode.test.fields.texts"
                            components={{ field: <strong>texts</strong> }}
                          />
                        </div>
                        <div>
                          <Trans
                            i18nKey="guardrails.customCode.test.fields.images"
                            components={{ field: <strong>images</strong> }}
                          />
                        </div>
                        <div>
                          <Trans
                            i18nKey="guardrails.customCode.test.fields.tools"
                            components={{
                              field: <strong>tools</strong>,
                              preCall: <span className="text-warning">(pre_call)</span>,
                              preMcpCall: <span className="text-purple-600">(pre_mcp_call)</span>,
                            }}
                          />
                        </div>
                        <div>
                          <Trans
                            i18nKey="guardrails.customCode.test.fields.toolCalls"
                            components={{
                              field: <strong>tool_calls</strong>,
                              postCall: <span className="text-success">(post_call)</span>,
                            }}
                          />
                        </div>
                        <div>
                          <Trans
                            i18nKey="guardrails.customCode.test.fields.structuredMessages"
                            components={{ field: <strong>structured_messages</strong> }}
                          />{" "}
                          <span className="text-warning">(pre_call)</span>
                        </div>
                        <div>
                          <Trans
                            i18nKey="guardrails.customCode.test.fields.model"
                            components={{ field: <strong>model</strong> }}
                          />
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
                      {isTesting ? t("guardrails.customCode.test.running") : t("guardrails.customCode.test.run")}
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
                            <CheckCircle2 className="size-4" />
                            {` ${t("guardrails.customCode.test.result.allowed")}`}
                          </>
                        ) : testResult.action === "block" ? (
                          <>
                            <XCircle className="size-4" />
                            {` ${t("guardrails.customCode.test.result.blocked")} `}
                            {testResult.reason}
                          </>
                        ) : testResult.action === "modify" ? (
                          <>
                            <CheckCircle2 className="size-4" />
                            {` ${t("guardrails.customCode.test.result.modified")}`}
                            {testResult.texts && testResult.texts.length > 0 && (
                              <span className="ml-1 text-xs text-muted-foreground">
                                -&gt; {testResult.texts[0].substring(0, 50)}
                                {testResult.texts[0].length > 50 ? "..." : ""}
                              </span>
                            )}
                          </>
                        ) : (
                          <>
                            <CheckCircle2 className="size-4" />{" "}
                            {testResult.action || t("guardrails.customCode.test.result.unknownAction")}
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
                  <div className="text-sm font-medium">{t("guardrails.customCode.contribute.title")}</div>
                  <div className="text-xs text-muted-foreground">
                    {t("guardrails.customCode.contribute.description")}
                  </div>
                </div>
              </div>
              <Button size="sm" onClick={() => window.open("https://github.com/BerriAI/litellm-guardrails", "_blank")}>
                <ExternalLink />
                {t("guardrails.customCode.contribute.button")}
              </Button>
            </div>
          </div>

          {/* Primitives Panel */}
          <div className="w-[300px] shrink-0 overflow-auto border-l border-border pl-6">
            <div className="mb-3 flex items-center gap-2">
              <Code className="size-4 text-muted-foreground" />
              <span className="font-semibold">{t("guardrails.customCode.primitives.title")}</span>
            </div>
            <p className="mb-3 text-xs text-muted-foreground">{t("guardrails.customCode.primitives.hint")}</p>

            <div className="space-y-2">
              {PRIMITIVE_CATEGORIES.map(({ id, labelKey, primitives }) => (
                <Collapsible
                  key={id}
                  defaultOpen={id === DEFAULT_OPEN_PRIMITIVE_CATEGORY}
                  className="rounded-lg border border-border"
                >
                  <CollapsibleTrigger className="group flex w-full items-center justify-between px-3 py-2 text-sm font-medium">
                    {t(labelKey)}
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
                              <CheckCircle2 className="size-3.5" />
                              {` ${t("guardrails.customCode.primitives.copied")}`}
                            </span>
                          ) : (
                            <>
                              <div className="font-mono text-xs">{p.name}</div>
                              <div className="mt-0.5 text-[10px] text-muted-foreground">{t(p.descKey)}</div>
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
          <span className="text-xs text-muted-foreground">{t("guardrails.customCode.footer.autosaveNote")}</span>
          <div className="flex items-center gap-3">
            <Button variant="secondary" onClick={onClose}>
              {t("guardrails.customCode.footer.cancel")}
            </Button>
            <Button onClick={handleSave} disabled={isSaving || !guardrailName.trim()} aria-busy={isSaving}>
              {isSaving ? <UiLoadingSpinner className="size-4" /> : <Save />}
              {isEditMode ? t("guardrails.customCode.footer.update") : t("guardrails.customCode.footer.save")}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default CustomCodeModal;
