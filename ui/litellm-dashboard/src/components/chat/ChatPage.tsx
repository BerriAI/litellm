"use client";

import React, { useCallback, useEffect, useRef, useState, useLayoutEffect } from "react";
import { Tooltip, Skeleton, Popover, message } from "antd";
import {
  SettingOutlined,
  PlusOutlined,
  EditOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  SearchOutlined,
  MessageOutlined,
  AppstoreOutlined,
  ArrowLeftOutlined,
  DownOutlined,
  CloseOutlined,
  CheckOutlined,
} from "@ant-design/icons";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useRouter, useSearchParams } from "next/navigation";
import { useChatHistory } from "./useChatHistory";
import ConversationList from "./ConversationList";
import ChatMessages from "./ChatMessages";
import MCPConnectPicker from "./MCPConnectPicker";
import MCPAppsPanel from "./MCPAppsPanel";
import { fetchAvailableModels } from "../playground/llm_calls/fetch_models";
import { makeOpenAIChatCompletionRequest } from "../playground/llm_calls/chat_completion";
import { makeOpenAIResponsesRequest } from "../playground/llm_calls/responses_api";
import type { MCPEvent } from "./types";
import { getProxyBaseUrl } from "@/components/networking";
import { useUIConfig } from "@/app/(dashboard)/hooks/uiConfig/useUIConfig";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";

interface ChatPageProps {
  accessToken: string;
  userRole: string;
  userId: string;
  userEmail?: string;
}

const SUGGESTIONS = ["Write", "Learn", "Code", "Brainstorm"];
const MAX_COMPARISON_MODELS = 3;
const LOCALSTORAGE_MODEL_KEY = "litellm_chat_selected_models";

function getGreeting(): string {
  const h = new Date().getHours();
  if (h >= 5 && h < 12) return "Good morning";
  if (h >= 12 && h < 17) return "Good afternoon";
  return "Good evening";
}

// Build the chat UI URL respecting server root path (e.g. /api/v1/ui/chat)
function getChatUrl(root: string, id?: string): string {
  return id ? `${root}/ui/chat?id=${id}` : `${root}/ui/chat`;
}

// Build the dashboard root URL (e.g. /api/v1/ui/)
function getDashboardUrl(root: string): string {
  const base = process.env.NEXT_PUBLIC_BASE_URL ?? "";
  const trimmed = base.replace(/^\/+|\/+$/g, "");
  return trimmed ? `${root}/${trimmed}/` : `${root}/`;
}

// Extract provider from model name for logo lookup.
// Handles prefixed models ("groq/llama-3"), and detects well-known providers by keyword.
function getProviderFromModelName(modelName: string): string {
  if (!modelName) return "";
  const lower = modelName.toLowerCase();
  const slash = lower.indexOf("/");
  if (slash > 0) return lower.slice(0, slash);
  // Keyword matching — order matters (more specific first)
  if (lower.includes("claude")) return "anthropic";
  if (lower.includes("gemini")) return "gemini";
  if (lower.includes("gpt") || lower.includes("chatgpt") || /^o[0-9]/.test(lower)) return "openai";
  if (lower.includes("mistral") || lower.includes("codestral")) return "mistral";
  if (lower.includes("llama")) return "meta_llama";
  if (lower.includes("deepseek")) return "deepseek";
  if (lower.includes("grok")) return "xai";
  if (lower.includes("command")) return "cohere";
  if (lower.includes("nova") || lower.includes("titan")) return "bedrock";
  return "";
}

interface ComparisonExchange {
  userMessage: string;
  responses: Record<string, string>; // model → accumulated response text
}

// Module-level async helper — each model gets its own independent Promise so they all run in parallel.
async function streamToModel(
  model: string,
  messages: Array<{ role: "user" | "assistant"; content: string }>,
  accessToken: string,
  mcpServers: string[],
  signal: AbortSignal,
  onChunk: (model: string, chunk: string) => void,
  onDone: (model: string) => void,
): Promise<void> {
  try {
    await makeOpenAIChatCompletionRequest(
      messages,
      (chunk: string) => onChunk(model, chunk),
      model,
      accessToken,
      undefined, // tags
      signal,
      undefined, // onReasoningContent
      undefined, undefined, undefined, undefined, undefined, undefined, // positions 8-13
      mcpServers.length > 0 ? mcpServers : undefined, // position 14: selectedMCPServers
    );
  } catch (err: unknown) {
    // Surface real errors in the response card; ignore user-triggered aborts
    if (!(err instanceof Error && err.name === "AbortError")) {
      const msg = err instanceof Error ? err.message : String(err);
      onChunk(model, `\n\n_Error: ${msg}_`);
    }
  } finally {
    onDone(model);
  }
}

const ChatPage: React.FC<ChatPageProps> = ({ accessToken, userRole, userId, userEmail }) => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const activeConversationId = searchParams.get("id");
  const { data: uiConfig } = useUIConfig();
  const uiRoot = uiConfig?.server_root_path && uiConfig.server_root_path !== "/"
    ? uiConfig.server_root_path.replace(/\/+$/, "")
    : "";
  const logoSrc = `${getProxyBaseUrl()}/get_image`;

  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [modelSelectorOpen, setModelSelectorOpen] = useState(false);
  const [modelSearchText, setModelSearchText] = useState("");

  const [selectedMCPServers, setSelectedMCPServers] = useState<string[]>([]);
  const [responsesSessionId, setResponsesSessionId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [inputText, setInputText] = useState("");
  const [mcpPopoverOpen, setMcpPopoverOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarView, setSidebarView] = useState<"chats" | "apps">("chats");
  const [storageBannerDismissed, setStorageBannerDismissed] = useState(false);

  // Comparison mode state (active when selectedModels.length > 1)
  // Each exchange holds the user message + per-model responses so we can do multi-turn comparison.
  const [comparisonExchanges, setComparisonExchanges] = useState<ComparisonExchange[]>([]);
  const [comparisonStreamingSet, setComparisonStreamingSet] = useState<Set<string>>(new Set());
  const comparisonAbortControllersRef = useRef<Record<string, AbortController>>({});

  const abortControllerRef = useRef<AbortController | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesScrollRef = useRef<HTMLDivElement>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const streamScrollLock = useRef<number | null>(null);

  const {
    conversations,
    activeConversation,
    storageUnavailable,
    staleId,
    createConversation,
    appendMessage,
    updateLastAssistantMessage,
    truncateFromMessage,
    deleteConversation,
    renameConversation,
  } = useChatHistory(activeConversationId);

  // Load models
  useEffect(() => {
    if (!accessToken) return;
    setIsLoadingModels(true);
    fetchAvailableModels(accessToken)
      .then((data) => {
        const names = (data || []).map((m: { model_group?: string }) => m.model_group ?? "").filter(Boolean);
        setModels(names);
        try {
          const saved = localStorage.getItem(LOCALSTORAGE_MODEL_KEY);
          if (saved) {
            const parsed: unknown = JSON.parse(saved);
            if (Array.isArray(parsed)) {
              const valid = (parsed as string[]).filter((m) => names.includes(m));
              if (valid.length > 0) {
                setSelectedModels(valid);
                return;
              }
            }
          }
        } catch {
          // ignore parse errors
        }
        if (names.length > 0) {
          setSelectedModels([names[0]]);
          localStorage.setItem(LOCALSTORAGE_MODEL_KEY, JSON.stringify([names[0]]));
        }
      })
      .catch(() => message.error("Could not load models"))
      .finally(() => setIsLoadingModels(false));
  }, [accessToken]);

  useEffect(() => {
    if (staleId) router.replace(getChatUrl(uiRoot));
  }, [staleId, router]);

  // Reset the responses session when switching between conversations so that
  // previous_response_id from conversation A is never sent for conversation B.
  useEffect(() => {
    setResponsesSessionId(null);
  }, [activeConversationId]);

  const toggleModel = useCallback((model: string) => {
    setSelectedModels((prev) => {
      let next: string[];
      if (prev.includes(model)) {
        next = prev.filter((m) => m !== model);
      } else if (prev.length >= MAX_COMPARISON_MODELS) {
        return prev;
      } else {
        next = [...prev, model];
      }
      localStorage.setItem(LOCALSTORAGE_MODEL_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const isComparisonMode = selectedModels.length > 1;
  const isAnyStreaming = isStreaming || comparisonStreamingSet.size > 0;

  const handleSend = useCallback(
    async (text: string, historyOverride?: Array<{ role: "user" | "assistant"; content: string }>) => {
      const trimmed = text.trim();
      if (!trimmed || selectedModels.length === 0 || isStreaming) return;
      const model = selectedModels[0];
      setInputText("");

      let convId = activeConversationId;
      if (!convId) {
        convId = createConversation(model);
        setResponsesSessionId(null); // new conversation starts a fresh session
        router.push(getChatUrl(uiRoot, convId));
      }

      appendMessage(convId, { role: "user", content: trimmed });
      appendMessage(convId, { role: "assistant", content: "" });

      setIsStreaming(true);
      abortControllerRef.current = new AbortController();

      // When historyOverride is set (edit / retry), the existing server-side
      // session chain covers messages that were just truncated and is no longer
      // valid for the rewritten history.  Eagerly clear the session so that a
      // failed/aborted edit does not leave a stale session ID that contaminates
      // the next regular send.
      if (historyOverride) {
        setResponsesSessionId(null);
      }

      // On a normal continuation turn with an active session, the Responses API
      // already holds the prior context server-side, so we only pass the new
      // user message (sending the full history would double-count it).
      //
      // On the very first turn (no session yet), we send the full history.
      const previousResponseId = historyOverride ? null : responsesSessionId;

      const history: Array<{ role: "user" | "assistant"; content: string }> =
        historyOverride
          ? [...historyOverride, { role: "user" as const, content: trimmed }]
          : previousResponseId
          ? [{ role: "user" as const, content: trimmed }]
          : [
              // Explicitly filter to only user/assistant roles — tool messages
              // lack a required tool_call_id and would cause API errors.
              ...(activeConversation?.messages ?? [])
                .filter((m): m is typeof m & { role: "user" | "assistant" } =>
                  m.role === "user" || m.role === "assistant"
                )
                .map((m) => ({ role: m.role, content: m.content })),
              { role: "user" as const, content: trimmed },
            ];

      let accumulatedContent = "";
      let accumulatedReasoning = "";
      // MCP events accumulated locally so we can persist them to the message
      // without relying on component state (which would cause stale closures).
      const accumulatedMCPEvents: MCPEvent[] = [];
      // Track clean completion so partial events are not shown on error/abort.
      let streamCompletedCleanly = false;

      try {
        await makeOpenAIResponsesRequest(
          history,
          (_role: string, chunk: string) => {
            accumulatedContent += chunk;
            updateLastAssistantMessage(convId!, { content: accumulatedContent });
          },
          model,
          accessToken,
          undefined, // tags
          abortControllerRef.current.signal,
          (rc: string) => {
            accumulatedReasoning += rc;
            updateLastAssistantMessage(convId!, { reasoningContent: accumulatedReasoning });
          },
          undefined, undefined, undefined, undefined, undefined, undefined,
          selectedMCPServers.length > 0 ? selectedMCPServers : undefined,
          previousResponseId,
          (id: string) => setResponsesSessionId(id),
          (event: MCPEvent) => {
            // Update in real-time so users see tool activity as it happens.
            // MCP events are infrequent (a handful per turn) so the extra
            // localStorage writes are acceptable.
            accumulatedMCPEvents.push(event);
            updateLastAssistantMessage(convId!, { mcpEvents: [...accumulatedMCPEvents] });
          },
        );
        streamCompletedCleanly = true;
      } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") {
          updateLastAssistantMessage(convId!, {
            content: accumulatedContent + " [stopped]",
          });
        } else {
          updateLastAssistantMessage(convId!, {
            content: "[Something went wrong. The partial response has been saved.]",
          });
        }
      } finally {
        // Clear partial MCP events on non-clean completion so users don't see
        // incomplete tool calls (e.g. a call_tool without its output).
        if (!streamCompletedCleanly && accumulatedMCPEvents.length > 0) {
          updateLastAssistantMessage(convId!, { mcpEvents: [] });
        }
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    },
    [activeConversationId, activeConversation, selectedModels, selectedMCPServers, accessToken,
      createConversation, appendMessage, updateLastAssistantMessage, router, isStreaming, responsesSessionId],
  );

  const handleSendComparison = useCallback(
    (text: string, currentExchanges: ComparisonExchange[]) => {
      const trimmed = text.trim();
      if (!trimmed || selectedModels.length === 0 || isAnyStreaming) return;
      setInputText("");

      // Append a new exchange with empty responses
      const newExchange: ComparisonExchange = { userMessage: trimmed, responses: {} };
      const newExchangeIdx = currentExchanges.length;
      setComparisonExchanges((prev) => [...prev, newExchange]);
      setComparisonStreamingSet(new Set(selectedModels));

      const controllers: Record<string, AbortController> = {};
      selectedModels.forEach((m) => { controllers[m] = new AbortController(); });
      comparisonAbortControllersRef.current = controllers;

      // Launch all model streams simultaneously — Promise.allSettled ensures they run in parallel
      // (streamToModel handles its own errors internally so these promises won't reject)
      void Promise.allSettled(
        selectedModels.map((model) => {
          // Build per-model history: each model's past responses become its own context
          const history: Array<{ role: "user" | "assistant"; content: string }> = [];
          for (const ex of currentExchanges) {
            history.push({ role: "user", content: ex.userMessage });
            history.push({ role: "assistant", content: ex.responses[model] ?? "" });
          }
          history.push({ role: "user", content: trimmed });

          return streamToModel(
            model,
            history,
            accessToken,
            selectedMCPServers,
            controllers[model].signal,
            (m, chunk) => setComparisonExchanges((prev) => {
              const updated = [...prev];
              const ex = { ...updated[newExchangeIdx] };
              ex.responses = { ...ex.responses, [m]: (ex.responses[m] ?? "") + chunk };
              updated[newExchangeIdx] = ex;
              return updated;
            }),
            (m) => setComparisonStreamingSet((prev) => { const next = new Set(prev); next.delete(m); return next; }),
          );
        }),
      );
    },
    [selectedModels, accessToken, selectedMCPServers, isAnyStreaming],
  );

  const handleStop = useCallback(() => {
    abortControllerRef.current?.abort();
    Object.values(comparisonAbortControllersRef.current).forEach((c) => c.abort());
    comparisonAbortControllersRef.current = {};
  }, []);

  const handleEditAndResend = useCallback(
    (messageId: string, newContent: string) => {
      if (!activeConversationId || isStreaming) return;
      // Compute the truncated history synchronously before the async state update lands,
      // so handleSend receives the correct pre-edit context rather than the stale closure value.
      const msgs = activeConversation?.messages ?? [];
      const idx = msgs.findIndex((m) => m.id === messageId);
      const priorMessages = (idx === -1 ? msgs : msgs.slice(0, idx))
        .filter((m) => m.role === "user" || m.role === "assistant")
        .map((m) => ({ role: m.role as "user" | "assistant", content: m.content }));
      truncateFromMessage(activeConversationId, messageId);
      handleSend(newContent, priorMessages);
    },
    [activeConversationId, isStreaming, activeConversation, truncateFromMessage, handleSend],
  );

  const handleSubmit = useCallback(
    (text: string) => {
      if (isComparisonMode) {
        handleSendComparison(text, comparisonExchanges);
      } else {
        handleSend(text);
      }
    },
    [isComparisonMode, handleSend, handleSendComparison, comparisonExchanges],
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(inputText);
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 180)}px`;
  }, [inputText]);

  // Track scroll position to show/hide scroll-to-bottom button
  useEffect(() => {
    const el = messagesScrollRef.current;
    if (!el) return;
    const onScroll = () => {
      const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
      setShowScrollButton(distFromBottom > 120);
      if (streamScrollLock.current !== null) {
        streamScrollLock.current = el.scrollTop; // track user-initiated scroll
      }
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [activeConversation]);

  // Start/stop the scroll lock when streaming begins/ends
  useEffect(() => {
    const el = messagesScrollRef.current;
    if (isStreaming) {
      streamScrollLock.current = el?.scrollTop ?? 0;
    } else {
      streamScrollLock.current = null;
    }
  }, [isStreaming]);

  // After every render during streaming, restore the locked scroll position
  useLayoutEffect(() => {
    if (streamScrollLock.current === null) return;
    const el = messagesScrollRef.current;
    if (!el) return;
    el.scrollTop = streamScrollLock.current;
  });

  // Scroll to bottom only when message COUNT increases (new message added)
  const prevMsgCountRef = useRef(0);
  useLayoutEffect(() => {
    const count = activeConversation?.messages?.length ?? 0;
    const prev = prevMsgCountRef.current;
    prevMsgCountRef.current = count;
    if (count > prev) {
      const el = messagesScrollRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    }
  }, [activeConversation?.messages]);

  const showBlankState = !isComparisonMode
    ? (!activeConversation || activeConversation.messages.length === 0)
    : comparisonExchanges.length === 0;
  const displayName = userEmail?.split("@")[0] ?? userId ?? "";
  const greeting = displayName ? `${getGreeting()}, ${displayName}` : getGreeting();
  const dashboardUrl = getDashboardUrl(uiRoot);

  // Filtered models: selected ones float to the top, then alphabetical
  const filteredModels = (modelSearchText
    ? models.filter((m) => m.toLowerCase().includes(modelSearchText.toLowerCase()))
    : models
  ).sort((a, b) => {
    const aSelected = selectedModels.includes(a);
    const bSelected = selectedModels.includes(b);
    if (aSelected && !bSelected) return -1;
    if (!aSelected && bSelected) return 1;
    return 0;
  });

  // ---- Model selector popover content ----
  const modelSelectorContent = (
    <div style={{ width: 280, maxHeight: 400, display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "8px 8px 4px" }}>
        <input
          autoFocus
          value={modelSearchText}
          onChange={(e) => setModelSearchText(e.target.value)}
          placeholder="Search models..."
          style={{
            width: "100%",
            padding: "6px 10px",
            border: "1px solid #d1d5db",
            borderRadius: 6,
            fontSize: 13,
            outline: "none",
            boxSizing: "border-box",
          }}
        />
      </div>
      {selectedModels.length >= MAX_COMPARISON_MODELS && (
        <div style={{ padding: "4px 12px", fontSize: 12, color: "#6b7280" }}>
          Max {MAX_COMPARISON_MODELS} models selected — deselect one to change.
        </div>
      )}
      <div style={{ flex: 1, overflowY: "auto" }}>
        {filteredModels.map((m) => {
          const checked = selectedModels.includes(m);
          const disabled = !checked && selectedModels.length >= MAX_COMPARISON_MODELS;
          const provider = getProviderFromModelName(m);
          const { logo } = provider ? getProviderLogoAndName(provider) : { logo: "" };
          return (
            <button
              key={m}
              disabled={disabled}
              onClick={() => toggleModel(m)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                width: "100%",
                padding: "7px 12px",
                background: checked ? "#eff6ff" : "transparent",
                border: "none",
                cursor: disabled ? "not-allowed" : "pointer",
                textAlign: "left",
                opacity: disabled ? 0.45 : 1,
                borderRadius: 4,
              }}
            >
              <span style={{
                width: 16, height: 16, borderRadius: 3, border: `1.5px solid ${checked ? "#1677ff" : "#d1d5db"}`,
                background: checked ? "#1677ff" : "#fff",
                display: "flex", alignItems: "center", justifyContent: "center",
                flexShrink: 0, transition: "all 0.1s",
              }}>
                {checked && <CheckOutlined style={{ fontSize: 10, color: "#fff" }} />}
              </span>
              {logo ? (
                <img src={logo} alt="" style={{ width: 16, height: 16, objectFit: "contain", flexShrink: 0 }} onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }} />
              ) : (
                <span style={{ width: 16, flexShrink: 0 }} />
              )}
              <span style={{ fontSize: 13, color: "#111827", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {m}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );

  // ---- Sidebar nav item renderer ----
  const sidebarNavItem = (
    icon: React.ReactNode,
    label: string,
    onClick: () => void,
    active = false,
    kbd?: string,
  ) => (
    <Tooltip title={sidebarCollapsed ? label : undefined} placement="right" key={label}>
      <button
        onClick={onClick}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "8px 10px",
          width: "100%",
          borderRadius: 7,
          border: "none",
          cursor: "pointer",
          background: active ? "#e8f4ff" : "transparent",
          color: active ? "#1677ff" : "#374151",
          textAlign: "left",
          fontSize: 14,
          justifyContent: sidebarCollapsed ? "center" : "flex-start",
          transition: "background 0.12s",
        }}
        onMouseEnter={(e) => {
          if (!active) (e.currentTarget as HTMLButtonElement).style.background = "#f5f5f5";
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLButtonElement).style.background = active ? "#e8f4ff" : "transparent";
        }}
      >
        <span style={{ fontSize: 16, flexShrink: 0 }}>{icon}</span>
        {!sidebarCollapsed && (
          <>
            <span style={{ flex: 1 }}>{label}</span>
            {kbd && <span style={{ fontSize: 11, color: "#9ca3af" }}>{kbd}</span>}
          </>
        )}
      </button>
    </Tooltip>
  );

  // ---- Model selector trigger button ----
  const modelSelectorTrigger = isLoadingModels ? (
    <Skeleton.Input active style={{ width: 160, height: 28 }} />
  ) : (
    <Popover
      open={modelSelectorOpen}
      onOpenChange={(open) => { setModelSelectorOpen(open); if (!open) setModelSearchText(""); }}
      content={modelSelectorContent}
      trigger="click"
      placement="bottomLeft"
    >
      <button
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          padding: "5px 10px",
          borderRadius: 7,
          border: "1px solid transparent",
          cursor: "pointer",
          background: "transparent",
          color: "#111827",
          fontSize: 14,
          fontWeight: 500,
          maxWidth: 480,
          overflow: "hidden",
        }}
        onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "#f5f5f5"; }}
        onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "transparent"; }}
      >
        {selectedModels.length === 0 ? (
          <span style={{ color: "#9ca3af" }}>Select model</span>
        ) : selectedModels.length === 1 ? (
          <>
            {(() => {
              const provider = getProviderFromModelName(selectedModels[0]);
              const { logo } = provider ? getProviderLogoAndName(provider) : { logo: "" };
              return logo ? <img src={logo} alt="" style={{ width: 18, height: 18, objectFit: "contain", flexShrink: 0 }} onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }} /> : null;
            })()}
            <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 240 }}>
              {selectedModels[0]}
            </span>
          </>
        ) : (
          <div style={{ display: "flex", alignItems: "center", gap: 4, flexWrap: "nowrap", overflow: "hidden" }}>
            {selectedModels.map((m) => {
              const provider = getProviderFromModelName(m);
              const { logo } = provider ? getProviderLogoAndName(provider) : { logo: "" };
              return (
                <span key={m} style={{
                  display: "inline-flex", alignItems: "center", gap: 4,
                  padding: "2px 8px", background: "#f0f4ff", borderRadius: 10,
                  fontSize: 12, color: "#1677ff", fontWeight: 500, flexShrink: 0,
                }}>
                  {logo && <img src={logo} alt="" style={{ width: 13, height: 13, objectFit: "contain" }} onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }} />}
                  <span style={{ maxWidth: 120, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{m}</span>
                </span>
              );
            })}
          </div>
        )}
        <DownOutlined style={{ fontSize: 10, color: "#9ca3af", flexShrink: 0, marginLeft: 2 }} />
      </button>
    </Popover>
  );

  // ---- Shared input bar ----
  const inputBar = (inConversation: boolean) => (
    <div style={{
      background: "#fff",
      borderRadius: 12,
      border: "1px solid #e5e7eb",
      boxShadow: "0 1px 6px rgba(0,0,0,0.06)",
      overflow: "hidden",
    }}>
      <textarea
        ref={textareaRef}
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={inConversation ? "Send a message..." : "How can I help you today?"}
        style={{
          width: "100%",
          minHeight: inConversation ? 52 : 80,
          padding: inConversation ? "16px 20px 8px" : "20px 20px 8px",
          border: "none",
          outline: "none",
          resize: "none",
          fontSize: 15,
          color: "#111827",
          background: "transparent",
          fontFamily: "inherit",
          boxSizing: "border-box",
        }}
      />
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: inConversation ? "4px 12px 10px" : "8px 12px 12px",
        borderTop: "1px solid #f3f4f6",
      }}>
        <Popover
          open={mcpPopoverOpen}
          onOpenChange={setMcpPopoverOpen}
          content={
            <MCPConnectPicker
              accessToken={accessToken}
              selectedServers={selectedMCPServers}
              onChange={setSelectedMCPServers}
            />
          }
          trigger="click"
          placement="topLeft"
        >
          <button style={{
            background: "none", border: "1px solid #d1d5db",
            borderRadius: 6, padding: "5px 10px",
            cursor: "pointer", fontSize: 14, color: "#6b7280",
            display: "flex", alignItems: "center", gap: 4,
          }}>
            <PlusOutlined />
            {selectedMCPServers.length > 0 && (
              <span style={{ fontSize: 12, color: "#1677ff", fontWeight: 500 }}>
                {selectedMCPServers.length}
              </span>
            )}
          </button>
        </Popover>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {!isComparisonMode && (
            <span style={{ fontSize: 12, color: "#9ca3af", maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {inConversation
                ? (selectedMCPServers.length > 0 ? `${selectedMCPServers.length} tool${selectedMCPServers.length > 1 ? "s" : ""} connected` : "")
                : (selectedModels[0] || "No model")}
            </span>
          )}
          {isAnyStreaming ? (
            <button onClick={handleStop} style={{
              background: "none", border: "1.5px solid #d1d5db", borderRadius: "50%",
              width: 32, height: 32, cursor: "pointer", color: "#374151",
              display: "flex", alignItems: "center", justifyContent: "center",
              flexShrink: 0, transition: "border-color 0.15s",
            }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = "#9ca3af"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.borderColor = "#d1d5db"; }}
            >
              <div style={{ width: 10, height: 10, background: "#374151", borderRadius: 2 }} />
            </button>
          ) : (
            <button
              onClick={() => handleSubmit(inputText)}
              disabled={!inputText.trim() || isLoadingModels || selectedModels.length === 0}
              style={{
                background: inputText.trim() && selectedModels.length > 0 ? "#1677ff" : "#f3f4f6",
                border: "none", borderRadius: 7,
                padding: "7px 16px",
                cursor: inputText.trim() && selectedModels.length > 0 ? "pointer" : "not-allowed",
                color: inputText.trim() && selectedModels.length > 0 ? "#fff" : "#9ca3af",
                fontSize: 14, fontWeight: 500,
                transition: "background 0.15s",
              }}
            >
              Send
            </button>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div style={{
      display: "flex",
      height: "100vh",
      width: "100vw",
      background: "#ffffff",
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      overflow: "hidden",
    }}>

      {/* ===== LEFT SIDEBAR ===== */}
      <div style={{
        width: sidebarCollapsed ? 56 : 260,
        flexShrink: 0,
        background: "#f9fafb",
        borderRight: "1px solid #e5e7eb",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        transition: "width 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
      }}>

        {/* Sidebar header: logo + collapse button */}
        <div style={{
          display: "flex",
          alignItems: "center",
          padding: "12px 10px",
          justifyContent: sidebarCollapsed ? "center" : "space-between",
          flexShrink: 0,
        }}>
          {!sidebarCollapsed && (
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <img
                src={logoSrc}
                alt="LiteLLM"
                style={{ height: 28, maxWidth: 120, objectFit: "contain", flexShrink: 0 }}
              />
              <span style={{ fontWeight: 700, fontSize: 15, color: "#111827", letterSpacing: "-0.01em" }}>
                LiteLLM
              </span>
            </div>
          )}
          <Tooltip title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"} placement="right">
            <button
              onClick={() => setSidebarCollapsed((v) => !v)}
              style={{
                background: "none", border: "none", cursor: "pointer",
                padding: 6, borderRadius: 7, color: "#6b7280", fontSize: 16,
                display: "flex", alignItems: "center",
              }}
            >
              {sidebarCollapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            </button>
          </Tooltip>
        </div>

        {/* Sidebar nav buttons */}
        <div style={{ padding: "0 8px 4px", flexShrink: 0 }}>
          {sidebarNavItem(<EditOutlined />, "New chat", () => router.push(getChatUrl(uiRoot)))}
          {sidebarNavItem(<SearchOutlined />, "Search chats", () => setSidebarView("chats"))}
        </div>

        <div style={{ height: 1, background: "#e5e7eb", margin: "4px 8px", flexShrink: 0 }} />

        {/* Chats / Apps tabs + Back to console */}
        <div style={{ padding: "4px 8px", flexShrink: 0 }}>
          {sidebarNavItem(<MessageOutlined />, "Chats", () => setSidebarView("chats"), sidebarView === "chats")}
          {sidebarNavItem(<AppstoreOutlined />, "Apps", () => setSidebarView("apps"), sidebarView === "apps")}
          <Tooltip title={sidebarCollapsed ? "Back to Developer Console UI" : undefined} placement="right">
            <a
              href={dashboardUrl}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "8px 10px",
                width: "100%",
                borderRadius: 7,
                color: "#6b7280",
                textDecoration: "none",
                fontSize: 14,
                justifyContent: sidebarCollapsed ? "center" : "flex-start",
                boxSizing: "border-box",
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLAnchorElement).style.background = "#f5f5f5";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLAnchorElement).style.background = "transparent";
              }}
            >
              <ArrowLeftOutlined style={{ fontSize: 16, flexShrink: 0 }} />
              {!sidebarCollapsed && (
                <span>Back to Developer Console UI</span>
              )}
            </a>
          </Tooltip>
        </div>

        <div style={{ height: 1, background: "#e5e7eb", margin: "4px 8px", flexShrink: 0 }} />

        {/* Sidebar content — only conversation list, only when in chats view and expanded */}
        {!sidebarCollapsed && sidebarView === "chats" && (
          <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
            <ConversationList
              conversations={conversations}
              activeConversationId={activeConversationId}
              onSelect={(id) => router.push(getChatUrl(uiRoot, id))}
              onDelete={deleteConversation}
              onNewChat={() => router.push(getChatUrl(uiRoot))}
              onRename={renameConversation}
            />
          </div>
        )}

      </div>

      {/* ===== MAIN AREA ===== */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>

        {/* Top bar */}
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "8px 16px",
          flexShrink: 0,
          borderBottom: "1px solid #f0f0f0",
          background: "#fff",
          height: 48,
        }}>
          {/* Left: model selector */}
          <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0, flex: 1 }}>
            {modelSelectorTrigger}
          </div>

          {/* Right: settings */}
          <div style={{ display: "flex", alignItems: "center", gap: 4, flexShrink: 0 }}>
            <Tooltip title="Settings">
              <button style={{
                background: "none", border: "none", cursor: "pointer",
                padding: 7, borderRadius: 7, color: "#6b7280", fontSize: 16,
                display: "flex", alignItems: "center",
              }}>
                <SettingOutlined />
              </button>
            </Tooltip>
          </div>
        </div>

        {/* Storage warning banner */}
        {storageUnavailable && !storageBannerDismissed && (
          <div style={{
            background: "#fffbe6", borderBottom: "1px solid #ffe58f",
            padding: "6px 20px", fontSize: 13, color: "#874d00",
            display: "flex", justifyContent: "space-between", alignItems: "center",
          }}>
            <span>Chat history won&apos;t be saved in this browser session.</span>
            <button onClick={() => setStorageBannerDismissed(true)}
              style={{ background: "none", border: "none", cursor: "pointer", fontSize: 16, color: "#874d00" }}>
              ×
            </button>
          </div>
        )}

        {/* Content area */}
        <div style={{ flex: 1, minHeight: 0, overflow: "hidden", display: "flex", flexDirection: "column", background: "#fff" }}>

          {/* ---- Apps page view ---- */}
          {sidebarView === "apps" ? (
            <div style={{ flex: 1, minHeight: 0, overflow: "auto", maxWidth: 800, margin: "0 auto", width: "100%", padding: "32px 24px" }}>
              <MCPAppsPanel
                accessToken={accessToken}
                selectedServers={selectedMCPServers}
                onChange={setSelectedMCPServers}
              />
            </div>

          ) : showBlankState ? (
            /* ---- Blank state ---- */
            <div style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              padding: "0 24px 80px",
            }}>
              {/* Greeting */}
              <h1 style={{
                margin: "0 0 32px",
                fontSize: 28,
                fontWeight: 600,
                color: "#111827",
                fontFamily: "inherit",
                letterSpacing: "-0.01em",
                textAlign: "center",
              }}>
                {isComparisonMode
                  ? `Compare ${selectedModels.length} models`
                  : greeting}
              </h1>

              {isComparisonMode ? (
                <p style={{ margin: "-16px 0 24px", fontSize: 14, color: "#6b7280", textAlign: "center" }}>
                  Send a message to see responses side-by-side
                </p>
              ) : (
                <p style={{ margin: "-16px 0 28px", fontSize: 14, color: "#6b7280", textAlign: "center", maxWidth: 520, lineHeight: 1.6 }}>
                  Chat with 100+ LLMs + MCP tools — authenticate once, use them here.{" "}
                  <button
                    onClick={() => setSidebarView("apps")}
                    style={{ background: "none", border: "none", cursor: "pointer", color: "#1677ff", fontSize: 14, padding: 0, fontWeight: 500 }}
                  >
                    Open Apps →
                  </button>
                </p>
              )}

              {/* Input card */}
              <div style={{ width: "100%", maxWidth: 680 }}>
                {inputBar(false)}
              </div>

              {/* Suggestion chips — only in single-model mode */}
              {!isComparisonMode && (
                <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap", justifyContent: "center" }}>
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      onClick={() => setInputText(s + ": ")}
                      style={{
                        background: "#f9fafb",
                        border: "1px solid #e5e7eb",
                        borderRadius: 20,
                        padding: "7px 16px",
                        fontSize: 14,
                        color: "#374151",
                        cursor: "pointer",
                      }}
                      onMouseEnter={(e) => {
                        (e.currentTarget as HTMLButtonElement).style.background = "#f3f4f6";
                      }}
                      onMouseLeave={(e) => {
                        (e.currentTarget as HTMLButtonElement).style.background = "#f9fafb";
                      }}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}

            </div>
          ) : (
            /* ---- Active conversation (single-model or comparison) ---- */
            <div style={{
              flex: 1, minHeight: 0, display: "flex", flexDirection: "column",
              maxWidth: isComparisonMode ? (selectedModels.length >= 3 ? 1200 : 960) : 760,
              margin: "0 auto", width: "100%", padding: "0 24px", position: "relative",
            }}>
              <div ref={messagesScrollRef} style={{ flex: 1, minHeight: 0, overflow: "auto", paddingTop: 24, overflowAnchor: "none" }}>
                {isComparisonMode ? (
                  /* Comparison: multi-turn exchanges, each with user bubble + side-by-side response cards */
                  <div style={{ paddingBottom: 8 }}>
                    {comparisonExchanges.map((exchange, exchangeIdx) => {
                      const isLastExchange = exchangeIdx === comparisonExchanges.length - 1;
                      return (
                        <div key={exchangeIdx} style={{ marginBottom: 32 }}>
                          {/* User message bubble */}
                          <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 20 }}>
                            <div style={{
                              background: "#f3f4f6",
                              borderRadius: 16,
                              padding: "10px 16px",
                              maxWidth: "75%",
                              fontSize: 14,
                              color: "#111827",
                              lineHeight: 1.5,
                            }}>
                              {exchange.userMessage}
                            </div>
                          </div>

                          {/* Response cards side-by-side */}
                          <div style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
                            {selectedModels.map((model, idx) => {
                              const provider = getProviderFromModelName(model);
                              const { logo } = provider ? getProviderLogoAndName(provider) : { logo: "" };
                              const responseText = exchange.responses[model] ?? "";
                              const isModelStreaming = isLastExchange && comparisonStreamingSet.has(model);
                              return (
                                <div key={model} style={{
                                  flex: 1,
                                  border: "1px solid #e5e7eb",
                                  borderRadius: 12,
                                  overflow: "hidden",
                                  minWidth: 0,
                                }}>
                                  {/* Card header — only show on first exchange */}
                                  {exchangeIdx === 0 && (
                                    <div style={{
                                      padding: "10px 14px",
                                      borderBottom: "1px solid #f0f0f0",
                                      display: "flex",
                                      alignItems: "center",
                                      gap: 8,
                                      background: "#fafafa",
                                    }}>
                                      {logo ? (
                                        <img
                                          src={logo}
                                          alt=""
                                          style={{ width: 18, height: 18, objectFit: "contain", flexShrink: 0 }}
                                          onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
                                        />
                                      ) : (
                                        <div style={{ width: 18, height: 18, borderRadius: "50%", background: "#e5e7eb", flexShrink: 0 }} />
                                      )}
                                      <span style={{ fontWeight: 600, fontSize: 12, color: "#374151" }}>
                                        Response {idx + 1}
                                      </span>
                                      <span style={{
                                        fontSize: 11, color: "#9ca3af",
                                        overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                                        flex: 1, minWidth: 0,
                                      }}>
                                        {model}
                                      </span>
                                    </div>
                                  )}
                                  {/* Card body */}
                                  <div style={{ padding: "14px 16px", minHeight: 60, position: "relative" }}>
                                    {isModelStreaming && (
                                      <span style={{
                                        position: "absolute", top: 10, right: 12,
                                        fontSize: 9, color: "#1677ff",
                                      }}>●</span>
                                    )}
                                    {responseText ? (
                                      <ReactMarkdown
                                        remarkPlugins={[remarkGfm]}
                                        components={{
                                          p: ({ children }) => <p style={{ margin: "0 0 10px", lineHeight: 1.6, fontSize: 14, color: "#111827" }}>{children}</p>,
                                          code: ({ className, children }) => {
                                            const isBlock = /language-(\w+)/.exec(className || "");
                                            if (isBlock) {
                                              return (
                                                <pre style={{ background: "#f8f9fa", padding: "10px 12px", borderRadius: 6, overflow: "auto", fontSize: 13, margin: "8px 0" }}>
                                                  <code>{children}</code>
                                                </pre>
                                              );
                                            }
                                            return <code style={{ background: "#f3f4f6", padding: "2px 5px", borderRadius: 3, fontSize: 13 }}>{children}</code>;
                                          },
                                        }}
                                      >
                                        {responseText}
                                      </ReactMarkdown>
                                    ) : isModelStreaming ? (
                                      <span style={{ color: "#9ca3af", fontSize: 14 }}>Generating…</span>
                                    ) : (
                                      <span style={{ color: "#9ca3af", fontSize: 14 }}>—</span>
                                    )}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <ChatMessages
                    messages={activeConversation!.messages}
                    isStreaming={isStreaming}
                    onEditMessage={handleEditAndResend}
                  />
                )}
              </div>
              {showScrollButton && (
                <button
                  onClick={() => {
                    const el = messagesScrollRef.current;
                    if (el) {
                      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
                      if (streamScrollLock.current !== null) {
                        streamScrollLock.current = el.scrollHeight;
                      }
                    }
                  }}
                  style={{
                    position: "absolute",
                    bottom: 100,
                    left: "50%",
                    transform: "translateX(-50%)",
                    width: 34,
                    height: 34,
                    borderRadius: "50%",
                    background: "rgba(255,255,255,0.75)",
                    backdropFilter: "blur(6px)",
                    WebkitBackdropFilter: "blur(6px)",
                    border: "1px solid rgba(0,0,0,0.1)",
                    boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "#6b7280",
                    zIndex: 10,
                    transition: "background 0.15s",
                  }}
                  onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,255,255,0.95)"; }}
                  onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.background = "rgba(255,255,255,0.75)"; }}
                  aria-label="Scroll to bottom"
                >
                  <DownOutlined style={{ fontSize: 12 }} />
                </button>
              )}

              {/* Input bar (in conversation) */}
              <div style={{ padding: "12px 0 24px" }}>
                {inputBar(true)}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatPage;
