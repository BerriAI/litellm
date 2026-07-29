"use client";

import React, { useCallback, useEffect, useRef, useState, useLayoutEffect } from "react";
import { Plus, ChevronDown, Check, X } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import MessageManager from "@/components/molecules/message_manager";
import { useRouter } from "next/navigation";
import { useChatShell } from "@/contexts/ChatShellContext";
import { getChatRoutes } from "@/components/chat/ChatShell";
import ChatMessages from "@/components/chat/ChatMessages";
import MCPConnectPicker from "@/components/chat/MCPConnectPicker";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";
import { makeOpenAIResponsesRequest } from "@/components/llm_calls/responses_api";
import type { MCPEvent } from "@/components/chat/types";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";

const SUGGESTIONS = ["Write", "Learn", "Code", "Brainstorm"];
const LOCALSTORAGE_MODEL_KEY = "litellm_chat_selected_model";

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

  const [selectedModel, setSelectedModel] = useState<string | null>(null);
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

  const abortControllerRef = useRef<AbortController | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const messagesScrollRef = useRef<HTMLDivElement>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);
  const streamScrollLock = useRef<number | null>(null);

  useEffect(() => {
    if (staleId) router.replace(getChatRoutes().chats);
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
          if (saved && names.includes(saved)) {
            setSelectedModel(saved);
            return;
          }
        } catch {
          // ignore parse errors
        }
        if (names.length > 0) {
          setSelectedModel(names[0]);
          localStorage.setItem(LOCALSTORAGE_MODEL_KEY, names[0]);
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

  const selectModel = useCallback((model: string) => {
    setSelectedModel(model);
    localStorage.setItem(LOCALSTORAGE_MODEL_KEY, model);
    setModelSelectorOpen(false);
    setModelSearchText("");
  }, []);

  const handleSend = useCallback(
    async (text: string, historyOverride?: Array<{ role: "user" | "assistant"; content: string }>) => {
      const trimmed = text.trim();
      if (!trimmed || !selectedModel || isStreaming) return;
      const model = selectedModel;
      setInputText("");

      let convId = activeConversationId;
      if (!convId) {
        convId = createConversation(model);
        setResponsesSessionId(null); // new conversation starts a fresh session
        window.history.pushState(null, "", `${window.location.pathname}?id=${convId}`);
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
      selectedModel,
      selectedMCPServers,
      accessToken,
      createConversation,
      appendMessage,
      updateLastAssistantMessage,
      isStreaming,
      responsesSessionId,
    ],
  );

  const handleStop = useCallback(() => {
    abortControllerRef.current?.abort();
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

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend(inputText);
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

  const showBlankState = !activeConversation || activeConversation.messages.length === 0;
  const displayName = userEmail?.split("@")[0] ?? userId ?? "";
  const greeting = displayName ? `${getGreeting()}, ${displayName}` : getGreeting();

  // Filtered models: selected one floats to the top, then alphabetical
  const filteredModels = (
    modelSearchText ? models.filter((m) => m.toLowerCase().includes(modelSearchText.toLowerCase())) : models
  ).sort((a, b) => {
    if (a === selectedModel) return -1;
    if (b === selectedModel) return 1;
    return 0;
  });

  const modelSelectorContent = (
    <div className="w-[280px] h-[400px] flex flex-col overflow-hidden">
      <div className="p-2 pb-1">
        <Input
          autoFocus
          value={modelSearchText}
          onChange={(e) => setModelSearchText(e.target.value)}
          placeholder="Search models..."
          className="h-8 text-[13px]"
        />
      </div>
      <ScrollArea className="flex-1 h-0">
        {filteredModels.map((m) => {
          const checked = m === selectedModel;
          const provider = getProviderFromModelName(m);
          const { logo } = provider ? getProviderLogoAndName(provider) : { logo: "" };
          return (
            <Button
              key={m}
              variant="ghost"
              onClick={() => selectModel(m)}
              className={`h-auto w-full justify-start gap-2 rounded px-3 py-[7px] font-normal ${checked ? "bg-accent" : ""}`}
            >
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
              <span className="flex-1 text-left text-[13px] text-foreground overflow-hidden text-ellipsis whitespace-nowrap">
                {m}
              </span>
              {checked && <Check className="h-3.5 w-3.5 text-primary shrink-0" />}
            </Button>
          );
        })}
      </ScrollArea>
    </div>
  );

  const modelSelectorTrigger = isLoadingModels ? (
    <Skeleton className="w-40 h-8" />
  ) : (
    <Popover
      open={modelSelectorOpen}
      onOpenChange={(open) => {
        setModelSelectorOpen(open);
        if (!open) setModelSearchText("");
      }}
    >
      <PopoverTrigger
        render={
          <Button variant="outline" size="sm" className="max-w-[240px] justify-start gap-1.5 overflow-hidden">
            {selectedModel ? (
              <>
                {(() => {
                  const provider = getProviderFromModelName(selectedModel);
                  const { logo } = provider ? getProviderLogoAndName(provider) : { logo: "" };
                  return logo ? (
                    <img
                      src={logo}
                      alt=""
                      className="w-4 h-4 object-contain shrink-0"
                      onError={(e) => {
                        (e.currentTarget as HTMLImageElement).style.display = "none";
                      }}
                    />
                  ) : null;
                })()}
                <span className="overflow-hidden text-ellipsis whitespace-nowrap">{selectedModel}</span>
              </>
            ) : (
              <span className="text-muted-foreground">Select model</span>
            )}
            <ChevronDown className="h-3 w-3 text-muted-foreground shrink-0" />
          </Button>
        }
      />
      <PopoverContent align="start" side="top" className="p-0 w-auto">
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
        <div className="flex items-center gap-2 min-w-0">
          {modelSelectorTrigger}
          <Popover open={mcpPopoverOpen} onOpenChange={setMcpPopoverOpen}>
            <PopoverTrigger
              render={
                <Button variant="outline" size="sm" className="gap-1 px-2.5 text-muted-foreground">
                  <Plus className="h-3.5 w-3.5" />
                  {selectedMCPServers.length > 0 && (
                    <span className="text-xs text-primary font-medium">{selectedMCPServers.length}</span>
                  )}
                </Button>
              }
            />
            <PopoverContent side="top" align="start" className="p-0 w-auto">
              <MCPConnectPicker
                accessToken={accessToken}
                selectedServers={selectedMCPServers}
                onChange={setSelectedMCPServers}
              />
            </PopoverContent>
          </Popover>
        </div>

        <div className="flex items-center gap-2">
          {inConversation && selectedMCPServers.length > 0 && (
            <span className="text-xs text-muted-foreground max-w-[160px] overflow-hidden text-ellipsis whitespace-nowrap">
              {selectedMCPServers.length} tool{selectedMCPServers.length > 1 ? "s" : ""} connected
            </span>
          )}
          {isStreaming ? (
            <Button variant="outline" size="icon-sm" onClick={handleStop} className="rounded-full shrink-0">
              <div className="w-2.5 h-2.5 bg-foreground rounded-[2px]" />
            </Button>
          ) : (
            <Button
              size="sm"
              onClick={() => handleSend(inputText)}
              disabled={!inputText.trim() || isLoadingModels || !selectedModel}
            >
              Send
            </Button>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <>
      {storageUnavailable && !storageBannerDismissed && (
        <div className="bg-amber-50 border-b border-amber-200 px-5 py-1.5 text-[13px] text-amber-800 flex justify-between items-center">
          <span>Chat history won&apos;t be saved in this browser session</span>
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={() => setStorageBannerDismissed(true)}
            className="text-amber-800 hover:bg-amber-100 hover:text-amber-800"
          >
            <X className="size-3.5" />
          </Button>
        </div>
      )}

      <div className="flex-1 min-h-0 overflow-hidden flex flex-col bg-background">
        {showBlankState ? (
          <div className="flex-1 flex flex-col items-center justify-center px-6 pb-20">
            <h1 className="m-0 mb-8 text-[28px] font-semibold text-foreground tracking-tight text-center">
              {greeting}
            </h1>

            <p className="-mt-4 mb-7 text-sm text-muted-foreground text-center max-w-[520px] leading-relaxed">
              Chat with 100+ LLMs + MCP tools; authenticate once, use them here.{" "}
              <Button
                variant="link"
                onClick={() => router.push(getChatRoutes().integrations)}
                className="h-auto p-0 text-sm font-medium"
              >
                Open Integrations -&gt;
              </Button>
            </p>

            <div className="w-full max-w-[680px]">{inputBar(false)}</div>

            <div className="flex gap-2 mt-3.5 flex-wrap justify-center">
              {SUGGESTIONS.map((s) => (
                <Button
                  key={s}
                  variant="outline"
                  size="sm"
                  onClick={() => setInputText(s + ": ")}
                  className="rounded-full px-4 text-muted-foreground"
                >
                  {s}
                </Button>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex-1 min-h-0 flex flex-col mx-auto w-full px-6 relative" style={{ maxWidth: 760 }}>
            <div
              ref={messagesScrollRef}
              className="flex-1 min-h-0 overflow-auto pt-6"
              style={{ overflowAnchor: "none" }}
            >
              <ChatMessages
                messages={activeConversation!.messages}
                isStreaming={isStreaming}
                onEditMessage={handleEditAndResend}
              />
            </div>
            {showScrollButton && (
              <Button
                variant="ghost"
                size="icon"
                onClick={() => {
                  const el = messagesScrollRef.current;
                  if (el) {
                    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
                    if (streamScrollLock.current !== null) {
                      streamScrollLock.current = el.scrollHeight;
                    }
                  }
                }}
                className="absolute bottom-[100px] left-1/2 -translate-x-1/2 z-10 rounded-full border bg-background/75 text-muted-foreground shadow-sm backdrop-blur-md hover:bg-background/95 hover:text-muted-foreground"
                aria-label="Scroll to bottom"
              >
                <ChevronDown className="h-3 w-3" />
              </Button>
            )}

            <div className="py-3 pb-6">{inputBar(true)}</div>
          </div>
        )}
      </div>
    </>
  );
}
