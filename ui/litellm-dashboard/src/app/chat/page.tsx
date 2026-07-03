"use client";

import React, { useCallback, useEffect, useRef, useState, useLayoutEffect } from "react";
import { Plus, ChevronDown, Check } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import MessageManager from "@/components/molecules/message_manager";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useRouter } from "next/navigation";
import { useChatShell } from "@/contexts/ChatShellContext";
import { CHAT_ROUTES } from "@/components/chat/ChatShell";
import ChatMessages from "@/components/chat/ChatMessages";
import MCPConnectPicker from "@/components/chat/MCPConnectPicker";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";
import { makeOpenAIChatCompletionRequest } from "@/components/llm_calls/chat_completion";
import { makeOpenAIResponsesRequest } from "@/components/llm_calls/responses_api";
import type { MCPEvent } from "@/components/chat/types";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";

const SUGGESTIONS = ["Write", "Learn", "Code", "Brainstorm"];
const MAX_COMPARISON_MODELS = 3;
const LOCALSTORAGE_MODEL_KEY = "litellm_chat_selected_models";

function getGreeting(): string {
  const h = new Date().getHours();
  if (h >= 5 && h < 12) return "Good morning";
  if (h >= 12 && h < 17) return "Good afternoon";
  return "Good evening";
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

interface StreamToModelArgs {
  model: string;
  messages: Array<{ role: "user" | "assistant"; content: string }>;
  accessToken: string;
  mcpServers: string[];
  signal: AbortSignal;
  onChunk: (model: string, chunk: string) => void;
  onDone: (model: string) => void;
}

// Module-level async helper — each model gets its own independent Promise so they all run in parallel.
async function streamToModel({
  model,
  messages,
  accessToken,
  mcpServers,
  signal,
  onChunk,
  onDone,
}: StreamToModelArgs): Promise<void> {
  try {
    await makeOpenAIChatCompletionRequest(
      messages,
      (chunk: string) => onChunk(model, chunk),
      model,
      accessToken,
      undefined, // tags
      signal,
      undefined, // onReasoningContent
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined, // positions 8-13
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

export default function ChatConversationPage() {
  const router = useRouter();
  const {
    accessToken,
    userId,
    userEmail,
    selectedMCPServers,
    setSelectedMCPServers,
    activeConversationId,
    activeConversation,
    storageUnavailable,
    staleId,
    createConversation,
    appendMessage,
    updateLastAssistantMessage,
    truncateFromMessage,
  } = useChatShell();
  const hadActiveConversationOnMountRef = useRef(activeConversationId !== null);

  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [modelSelectorOpen, setModelSelectorOpen] = useState(false);
  const [modelSearchText, setModelSearchText] = useState("");

  const [responsesSessionId, setResponsesSessionId] = useState<string | null>(null);
  const [prevConversationIdForSessionReset, setPrevConversationIdForSessionReset] = useState(activeConversationId);
  const [isStreaming, setIsStreaming] = useState(false);
  const [inputText, setInputText] = useState("");
  const [mcpPopoverOpen, setMcpPopoverOpen] = useState(false);
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

  useEffect(() => {
    if (staleId) router.replace(CHAT_ROUTES.chats);
  }, [staleId, router]);

  // Load models
  useEffect(() => {
    if (!accessToken) return;
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
                setSelectedModels(hadActiveConversationOnMountRef.current ? [valid[0]] : valid);
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
      .catch(() => MessageManager.error("Could not load models"))
      .finally(() => setIsLoadingModels(false));
  }, [accessToken]);

  // Reset the responses session when switching between conversations so that
  // previous_response_id from conversation A is never sent for conversation B.
  if (activeConversationId !== prevConversationIdForSessionReset) {
    setPrevConversationIdForSessionReset(activeConversationId);
    setResponsesSessionId(null);
  }

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
        router.push(`${CHAT_ROUTES.chats}?id=${convId}`);
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

      const history: Array<{ role: "user" | "assistant"; content: string }> = historyOverride
        ? [...historyOverride, { role: "user" as const, content: trimmed }]
        : previousResponseId
          ? [{ role: "user" as const, content: trimmed }]
          : [
              // Explicitly filter to only user/assistant roles — tool messages
              // lack a required tool_call_id and would cause API errors.
              ...(activeConversation?.messages ?? [])
                .filter(
                  (m): m is typeof m & { role: "user" | "assistant" } => m.role === "user" || m.role === "assistant",
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
          undefined,
          undefined,
          undefined,
          undefined,
          undefined,
          undefined,
          selectedMCPServers.length > 0 ? selectedMCPServers : undefined,
          previousResponseId,
          (id: string) => setResponsesSessionId(id),
          (event: MCPEvent) => {
            // Accumulate locally only — persisted once in finally to avoid
            // one full localStorage write per MCP event during streaming.
            accumulatedMCPEvents.push(event);
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
        // Only persist MCP events on clean completion — partial events from an
        // aborted or errored turn would show incomplete tool calls to the user.
        if (accumulatedMCPEvents.length > 0 && streamCompletedCleanly) {
          updateLastAssistantMessage(convId!, { mcpEvents: accumulatedMCPEvents });
        }
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    },
    [
      activeConversationId,
      activeConversation,
      selectedModels,
      selectedMCPServers,
      accessToken,
      createConversation,
      appendMessage,
      updateLastAssistantMessage,
      router,
      isStreaming,
      responsesSessionId,
    ],
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
      selectedModels.forEach((m) => {
        controllers[m] = new AbortController();
      });
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

          return streamToModel({
            model,
            messages: history,
            accessToken,
            mcpServers: selectedMCPServers,
            signal: controllers[model].signal,
            onChunk: (m, chunk) =>
              setComparisonExchanges((prev) => {
                const updated = [...prev];
                const ex = { ...updated[newExchangeIdx] };
                ex.responses = { ...ex.responses, [m]: (ex.responses[m] ?? "") + chunk };
                updated[newExchangeIdx] = ex;
                return updated;
              }),
            onDone: (m) =>
              setComparisonStreamingSet((prev) => {
                const next = new Set(prev);
                next.delete(m);
                return next;
              }),
          });
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
    ? !activeConversation || activeConversation.messages.length === 0
    : comparisonExchanges.length === 0;
  const displayName = userEmail?.split("@")[0] ?? userId ?? "";
  const greeting = displayName ? `${getGreeting()}, ${displayName}` : getGreeting();

  // Filtered models: selected ones float to the top, then alphabetical
  const filteredModels = (
    modelSearchText ? models.filter((m) => m.toLowerCase().includes(modelSearchText.toLowerCase())) : models
  ).sort((a, b) => {
    const aSelected = selectedModels.includes(a);
    const bSelected = selectedModels.includes(b);
    if (aSelected && !bSelected) return -1;
    if (!aSelected && bSelected) return 1;
    return 0;
  });

  const modelSelectorContent = (
    <div className="w-[280px] max-h-[400px] flex flex-col overflow-hidden">
      <div className="p-2 pb-1">
        <input
          autoFocus
          value={modelSearchText}
          onChange={(e) => setModelSearchText(e.target.value)}
          placeholder="Search models..."
          className="w-full px-2.5 py-1.5 border rounded-md text-[13px] outline-none bg-background text-foreground"
        />
      </div>
      {selectedModels.length >= MAX_COMPARISON_MODELS && (
        <div className="px-3 py-1 text-xs text-muted-foreground">
          Max {MAX_COMPARISON_MODELS} models selected; deselect one to change
        </div>
      )}
      <ScrollArea className="flex-1 min-h-0">
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
              className={`flex items-center gap-2 w-full px-3 py-[7px] border-none text-left rounded transition-colors ${
                checked ? "bg-accent" : "bg-transparent hover:bg-accent/50"
              } ${disabled ? "opacity-45 cursor-not-allowed" : "cursor-pointer"}`}
            >
              <span
                className={`w-4 h-4 rounded-[3px] flex items-center justify-center shrink-0 transition-all ${
                  checked ? "bg-primary border-primary" : "bg-background border-border"
                }`}
                style={{ border: `1.5px solid ${checked ? "var(--color-primary)" : "var(--color-border)"}` }}
              >
                {checked && <Check className="h-2.5 w-2.5 text-primary-foreground" />}
              </span>
              {logo ? (
                <img
                  src={logo}
                  alt=""
                  className="w-4 h-4 object-contain shrink-0"
                  onError={(e) => {
                    (e.currentTarget as HTMLImageElement).style.display = "none";
                  }}
                />
              ) : (
                <span className="w-4 shrink-0" />
              )}
              <span className="text-[13px] text-foreground overflow-hidden text-ellipsis whitespace-nowrap">{m}</span>
            </button>
          );
        })}
      </ScrollArea>
    </div>
  );

  const modelSelectorTrigger = isLoadingModels ? (
    <Skeleton className="w-40 h-7" />
  ) : (
    <Popover
      open={modelSelectorOpen}
      onOpenChange={(open) => {
        setModelSelectorOpen(open);
        if (!open) setModelSearchText("");
      }}
    >
      <PopoverTrigger asChild>
        <button className="flex items-center gap-1.5 px-2.5 py-[5px] rounded-md border border-transparent cursor-pointer bg-transparent text-foreground text-sm font-medium max-w-[480px] overflow-hidden hover:bg-accent/50 transition-colors">
          {selectedModels.length === 0 ? (
            <span className="text-muted-foreground">Select model</span>
          ) : selectedModels.length === 1 ? (
            <>
              {(() => {
                const provider = getProviderFromModelName(selectedModels[0]);
                const { logo } = provider ? getProviderLogoAndName(provider) : { logo: "" };
                return logo ? (
                  <img
                    src={logo}
                    alt=""
                    className="w-[18px] h-[18px] object-contain shrink-0"
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).style.display = "none";
                    }}
                  />
                ) : null;
              })()}
              <span className="overflow-hidden text-ellipsis whitespace-nowrap max-w-[240px]">{selectedModels[0]}</span>
            </>
          ) : (
            <div className="flex items-center gap-1 flex-nowrap overflow-hidden">
              {selectedModels.map((m) => {
                const provider = getProviderFromModelName(m);
                const { logo } = provider ? getProviderLogoAndName(provider) : { logo: "" };
                return (
                  <span
                    key={m}
                    className="inline-flex items-center gap-1 px-2 py-0.5 bg-primary/10 rounded-[10px] text-xs text-primary font-medium shrink-0"
                  >
                    {logo && (
                      <img
                        src={logo}
                        alt=""
                        className="w-[13px] h-[13px] object-contain"
                        onError={(e) => {
                          (e.currentTarget as HTMLImageElement).style.display = "none";
                        }}
                      />
                    )}
                    <span className="max-w-[120px] overflow-hidden text-ellipsis whitespace-nowrap">{m}</span>
                  </span>
                );
              })}
            </div>
          )}
          <ChevronDown className="h-2.5 w-2.5 text-muted-foreground shrink-0 ml-0.5" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="p-0 w-auto">
        {modelSelectorContent}
      </PopoverContent>
    </Popover>
  );

  const inputBar = (inConversation: boolean) => (
    <div className="bg-background rounded-xl border shadow-[0_1px_6px_rgba(0,0,0,0.06)] overflow-hidden">
      <textarea
        ref={textareaRef}
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={inConversation ? "Send a message..." : "How can I help you today?"}
        className="w-full border-none outline-none resize-none text-[15px] text-foreground bg-transparent font-[inherit] box-border"
        style={{
          minHeight: inConversation ? 52 : 80,
          padding: inConversation ? "16px 20px 8px" : "20px 20px 8px",
        }}
      />
      <div
        className="flex items-center justify-between border-t"
        style={{ padding: inConversation ? "4px 12px 10px" : "8px 12px 12px" }}
      >
        <Popover open={mcpPopoverOpen} onOpenChange={setMcpPopoverOpen}>
          <PopoverTrigger asChild>
            <button className="border rounded-md px-2.5 py-[5px] cursor-pointer text-sm text-muted-foreground flex items-center gap-1 bg-transparent hover:bg-accent/50 transition-colors">
              <Plus className="h-3.5 w-3.5" />
              {selectedMCPServers.length > 0 && (
                <span className="text-xs text-primary font-medium">{selectedMCPServers.length}</span>
              )}
            </button>
          </PopoverTrigger>
          <PopoverContent side="top" align="start" className="p-0 w-auto">
            <MCPConnectPicker
              accessToken={accessToken}
              selectedServers={selectedMCPServers}
              onChange={setSelectedMCPServers}
            />
          </PopoverContent>
        </Popover>

        <div className="flex items-center gap-2">
          {!isComparisonMode && (
            <span className="text-xs text-muted-foreground max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap">
              {inConversation
                ? selectedMCPServers.length > 0
                  ? `${selectedMCPServers.length} tool${selectedMCPServers.length > 1 ? "s" : ""} connected`
                  : ""
                : selectedModels[0] || "No model"}
            </span>
          )}
          {isAnyStreaming ? (
            <button
              onClick={handleStop}
              className="w-8 h-8 rounded-full border-[1.5px] flex items-center justify-center shrink-0 cursor-pointer transition-colors hover:border-muted-foreground bg-transparent text-foreground"
            >
              <div className="w-2.5 h-2.5 bg-foreground rounded-[2px]" />
            </button>
          ) : (
            <button
              onClick={() => handleSubmit(inputText)}
              disabled={!inputText.trim() || isLoadingModels || selectedModels.length === 0}
              className={`border-none rounded-md px-4 py-[7px] text-sm font-medium transition-colors ${
                inputText.trim() && selectedModels.length > 0
                  ? "bg-primary text-primary-foreground cursor-pointer hover:bg-primary/90"
                  : "bg-muted text-muted-foreground cursor-not-allowed"
              }`}
            >
              Send
            </button>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <>
      <div className="flex items-center px-4 py-2 shrink-0 border-b bg-background h-12">
        <div className="flex items-center gap-2 min-w-0 flex-1">{modelSelectorTrigger}</div>
      </div>

      {storageUnavailable && !storageBannerDismissed && (
        <div className="bg-amber-50 border-b border-amber-200 px-5 py-1.5 text-[13px] text-amber-800 flex justify-between items-center">
          <span>Chat history won&apos;t be saved in this browser session</span>
          <button
            onClick={() => setStorageBannerDismissed(true)}
            className="text-amber-800 text-base cursor-pointer"
            style={{ background: "none", border: "none" }}
          >
            &times;
          </button>
        </div>
      )}

      <div className="flex-1 min-h-0 overflow-hidden flex flex-col bg-background">
        {showBlankState ? (
          <div className="flex-1 flex flex-col items-center justify-center px-6 pb-20">
            <h1 className="m-0 mb-8 text-[28px] font-semibold text-foreground tracking-tight text-center">
              {isComparisonMode ? `Compare ${selectedModels.length} models` : greeting}
            </h1>

            {isComparisonMode ? (
              <p className="-mt-4 mb-6 text-sm text-muted-foreground text-center">
                Send a message to see responses side-by-side
              </p>
            ) : (
              <p className="-mt-4 mb-7 text-sm text-muted-foreground text-center max-w-[520px] leading-relaxed">
                Chat with 100+ LLMs + MCP tools; authenticate once, use them here.{" "}
                <button
                  onClick={() => router.push(CHAT_ROUTES.integrations)}
                  className="text-primary text-sm font-medium cursor-pointer hover:underline"
                  style={{ background: "none", border: "none", padding: 0 }}
                >
                  Open Integrations -&gt;
                </button>
              </p>
            )}

            <div className="w-full max-w-[680px]">{inputBar(false)}</div>

            {!isComparisonMode && (
              <div className="flex gap-2 mt-3.5 flex-wrap justify-center">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    onClick={() => setInputText(s + ": ")}
                    className="bg-secondary border rounded-full px-4 py-[7px] text-sm text-foreground/70 cursor-pointer hover:bg-accent transition-colors"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div
            className="flex-1 min-h-0 flex flex-col mx-auto w-full px-6 relative"
            style={{ maxWidth: isComparisonMode ? (selectedModels.length >= 3 ? 1200 : 960) : 760 }}
          >
            <div
              ref={messagesScrollRef}
              className="flex-1 min-h-0 overflow-auto pt-6"
              style={{ overflowAnchor: "none" }}
            >
              {isComparisonMode ? (
                <div className="pb-2">
                  {comparisonExchanges.map((exchange, exchangeIdx) => {
                    const isLastExchange = exchangeIdx === comparisonExchanges.length - 1;
                    return (
                      <div key={exchangeIdx} className="mb-8">
                        <div className="flex justify-end mb-5">
                          <div className="bg-muted rounded-2xl px-4 py-2.5 max-w-[75%] text-sm text-foreground leading-relaxed">
                            {exchange.userMessage}
                          </div>
                        </div>

                        <div className="flex gap-3.5 items-start">
                          {selectedModels.map((model, idx) => {
                            const provider = getProviderFromModelName(model);
                            const { logo } = provider ? getProviderLogoAndName(provider) : { logo: "" };
                            const responseText = exchange.responses[model] ?? "";
                            const isModelStreaming = isLastExchange && comparisonStreamingSet.has(model);
                            return (
                              <div key={model} className="flex-1 border rounded-xl overflow-hidden min-w-0">
                                {exchangeIdx === 0 && (
                                  <div className="px-3.5 py-2.5 border-b flex items-center gap-2 bg-muted/50">
                                    {logo ? (
                                      <img
                                        src={logo}
                                        alt=""
                                        className="w-[18px] h-[18px] object-contain shrink-0"
                                        onError={(e) => {
                                          (e.currentTarget as HTMLImageElement).style.display = "none";
                                        }}
                                      />
                                    ) : (
                                      <div className="w-[18px] h-[18px] rounded-full bg-border shrink-0" />
                                    )}
                                    <span className="font-semibold text-xs text-foreground">Response {idx + 1}</span>
                                    <span className="text-[11px] text-muted-foreground overflow-hidden text-ellipsis whitespace-nowrap flex-1 min-w-0">
                                      {model}
                                    </span>
                                  </div>
                                )}
                                <div className="px-4 py-3.5 min-h-[60px] relative">
                                  {isModelStreaming && (
                                    <span className="absolute top-2.5 right-3 text-[9px] text-primary">&#9679;</span>
                                  )}
                                  {responseText ? (
                                    <ReactMarkdown
                                      remarkPlugins={[remarkGfm]}
                                      components={{
                                        p: ({ children }) => (
                                          <p className="mb-2.5 leading-relaxed text-sm text-foreground">{children}</p>
                                        ),
                                        code: ({ className, children }) => {
                                          const isBlock = /language-(\w+)/.exec(className || "");
                                          if (isBlock) {
                                            return (
                                              <pre className="bg-muted px-3 py-2.5 rounded-md overflow-auto text-[13px] my-2">
                                                <code>{children}</code>
                                              </pre>
                                            );
                                          }
                                          return (
                                            <code className="bg-muted px-1.5 py-0.5 rounded text-[13px]">
                                              {children}
                                            </code>
                                          );
                                        },
                                      }}
                                    >
                                      {responseText}
                                    </ReactMarkdown>
                                  ) : isModelStreaming ? (
                                    <span className="text-muted-foreground text-sm">Generating&#8230;</span>
                                  ) : (
                                    <span className="text-muted-foreground text-sm">&#8212;</span>
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
                className="absolute bottom-[100px] left-1/2 -translate-x-1/2 w-[34px] h-[34px] rounded-full bg-background/75 backdrop-blur-md border border-black/10 shadow-sm cursor-pointer flex items-center justify-center text-muted-foreground z-10 transition-colors hover:bg-background/95"
                aria-label="Scroll to bottom"
              >
                <ChevronDown className="h-3 w-3" />
              </button>
            )}

            <div className="py-3 pb-6">{inputBar(true)}</div>
          </div>
        )}
      </div>
    </>
  );
}
