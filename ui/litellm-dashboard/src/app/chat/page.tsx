"use client";

import React, { useCallback, useEffect, useRef, useState, useLayoutEffect } from "react";
import { Plus, ChevronDown, Paperclip, ImageIcon, X } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";
import { useRouter } from "next/navigation";
import { useChatShell } from "@/contexts/ChatShellContext";
import { getChatRoutes } from "@/components/chat/ChatShell";
import ChatMessages from "@/components/chat/ChatMessages";
import MCPConnectPicker from "@/components/chat/MCPConnectPicker";
import { fetchAvailableModels, type ModelGroup } from "@/components/llm_calls/fetch_models";
import type { AssistantMessageUpdate, ChatAttachment, ChatMessage } from "@/components/chat/types";
import ChatModelPicker, { isChatModel, modelChatMode, type ChatMode } from "@/components/chat/ChatModelPicker";
import MoyaiChatComposer from "@/components/chat/moyai/MoyaiChatComposer";
import MoyaiChatHome from "@/components/chat/moyai/MoyaiChatHome";
import { runChatRequest } from "@/components/chat/runChatRequest";
import { readChatAttachment, MAX_CHAT_ATTACHMENTS, CHAT_ATTACHMENT_ACCEPT } from "@/components/chat/attachmentHelpers";
import ChatFileChip from "@/components/chat/ChatFileChip";
import { useQueryState } from "nuqs";

const LOCALSTORAGE_MODEL_KEY = "litellm_chat_selected_model";

type ReplayMessage = Pick<ChatMessage, "role" | "content" | "attachments" | "generatedImages" | "mcpEvents">;

function hasMessageContent(text: string, images: ChatAttachment[], mode: ChatMode): boolean {
  return Boolean(text.trim()) || (mode === "chat" && images.length > 0);
}

function AttachmentPreview({
  attachments,
  disabled,
  onRemove,
}: {
  attachments: ChatAttachment[];
  disabled: boolean;
  onRemove: (id: string) => void;
}) {
  if (!attachments.length) return null;
  return (
    <div className="flex w-full flex-wrap gap-3 px-4 pt-4" aria-label="Attached files">
      {attachments.map((attachment) => (
        <div key={attachment.id} className={attachment.kind === "file" ? "relative max-w-64" : "relative w-20"}>
          {attachment.kind === "file" ? (
            <ChatFileChip file={attachment} />
          ) : (
            <>
              {/* User attachments use local data URLs, not the image optimizer. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={attachment.url} alt={attachment.name} className="h-20 w-20 rounded-xl border object-cover" />
              <p className="mt-1 truncate text-[11px] text-muted-foreground">{attachment.name}</p>
            </>
          )}
          <Button
            type="button"
            variant="secondary"
            size="icon-xs"
            aria-label={`Remove ${attachment.name}`}
            className="absolute -right-2 -top-2 rounded-full"
            disabled={disabled}
            onClick={() => onRemove(attachment.id)}
          >
            <X className="size-3" />
          </Button>
        </div>
      ))}
    </div>
  );
}

export default function ChatConversationPage() {
  const [, setConversationIdInUrl] = useQueryState("id", { history: "push" });
  const router = useRouter();
  const {
    accessToken,
    conversations,
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

  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [models, setModels] = useState<ModelGroup[]>([]);
  const [mode, setMode] = useState<ChatMode>("chat");
  const [attachments, setAttachments] = useState<ChatAttachment[]>([]);
  const [attachmentError, setAttachmentError] = useState<string | null>(null);
  const [isReadingAttachments, setIsReadingAttachments] = useState(false);
  const attachmentInputRef = useRef<HTMLInputElement>(null);
  const readAttachmentsRef = useRef(false);
  const draftVersionRef = useRef(0);
  const [isLoadingModels, setIsLoadingModels] = useState(true);

  const [responsesSessionId, setResponsesSessionId] = useState<string | null>(null);
  const responsesSessionGenerationRef = useRef(0);
  const previousResponsesConversationRef = useRef(activeConversationId);
  const [prevConversationIdForSessionReset, setPrevConversationIdForSessionReset] = useState(activeConversationId);
  const [isStreaming, setIsStreaming] = useState(false);
  const [inputText, setInputText] = useState("");
  const [mcpPopoverOpen, setMcpPopoverOpen] = useState(false);
  const [storageBannerDismissed, setStorageBannerDismissed] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  const requestConversationRef = useRef<string | null>(null);
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
        const available = (data || []).filter(isChatModel);
        const names = available.map((model) => model.model_group);
        setModels(available);
        try {
          const saved = localStorage.getItem(LOCALSTORAGE_MODEL_KEY);
          if (saved && names.includes(saved)) {
            setSelectedModel(saved);
            setMode(modelChatMode(available.find((model) => model.model_group === saved)!));
            return;
          }
        } catch {
          // ignore parse errors
        }
        if (names.length > 0) {
          setSelectedModel(names[0]);
          setMode(modelChatMode(available[0]));
        }
      })
      .catch(() => toast.error("Could not load models"))
      .finally(() => setIsLoadingModels(false));
  }, [accessToken]);

  useLayoutEffect(() => {
    if (
      previousResponsesConversationRef.current !== null &&
      previousResponsesConversationRef.current !== activeConversationId
    ) {
      responsesSessionGenerationRef.current += 1;
    }
    previousResponsesConversationRef.current = activeConversationId;
  }, [activeConversationId]);

  // Reset the responses session when switching between conversations so that
  // previous_response_id from conversation A is never sent for conversation B.
  if (activeConversationId !== prevConversationIdForSessionReset) {
    setPrevConversationIdForSessionReset(activeConversationId);
    setResponsesSessionId(null);
    setAttachments([]);
    setAttachmentError(null);
    setInputText("");
  }

  useEffect(() => {
    draftVersionRef.current += 1;
    if (abortControllerRef.current && requestConversationRef.current !== activeConversationId) {
      abortControllerRef.current.abort();
    }
  }, [activeConversationId]);

  const selectModel = useCallback(
    (model: ModelGroup) => {
      if (modelChatMode(model) === "image" && attachments.length > 0) {
        setAttachmentError("Remove the attached files before switching to image generation.");
        return;
      }
      responsesSessionGenerationRef.current += 1;
      setSelectedModel(model.model_group);
      setMode(modelChatMode(model));
      setResponsesSessionId(null);
      setAttachmentError(null);
      try {
        localStorage.setItem(LOCALSTORAGE_MODEL_KEY, model.model_group);
      } catch {
        return;
      }
    },
    [attachments.length],
  );

  const addAttachments = async (files: File[]) => {
    if (mode !== "chat" || files.length === 0) return;
    if (isStreaming || readAttachmentsRef.current) return;
    if (attachments.length + files.length > MAX_CHAT_ATTACHMENTS) {
      setAttachmentError(`Attach up to ${MAX_CHAT_ATTACHMENTS} files per message.`);
      return;
    }
    const draftVersion = draftVersionRef.current;
    readAttachmentsRef.current = true;
    setIsReadingAttachments(true);
    setAttachmentError(null);
    try {
      const images = await Promise.all(files.map(readChatAttachment));
      if (draftVersion === draftVersionRef.current) setAttachments((current) => [...current, ...images]);
    } catch (error) {
      if (draftVersion === draftVersionRef.current)
        setAttachmentError(error instanceof Error ? error.message : "Could not read these files.");
    } finally {
      readAttachmentsRef.current = false;
      setIsReadingAttachments(false);
    }
  };

  const handleAttachmentInput = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    void addAttachments(files);
  };

  useEffect(
    () => () => {
      abortControllerRef.current?.abort();
      draftVersionRef.current += 1;
    },
    [],
  );

  const handleSend = useCallback(
    async (
      text: string,
      historyOverride?: ReplayMessage[],
      attachmentOverride?: ChatAttachment[],
      generationModel?: string,
    ) => {
      const trimmed = text.trim();
      const images = attachmentOverride ?? attachments;
      const requestMode = generationModel ? "image" : mode;
      const model = generationModel ?? selectedModel;
      if (!hasMessageContent(trimmed, images, requestMode) || !model) return;
      if (isStreaming || abortControllerRef.current || readAttachmentsRef.current) return;
      const sessionGeneration = responsesSessionGenerationRef.current;
      setInputText("");

      let convId = activeConversationId;
      if (!convId) {
        convId = createConversation(model);
        setResponsesSessionId(null); // new conversation starts a fresh session
        setConversationIdInUrl(convId);
      }

      const imageGeneration = requestMode === "image" ? { model, prompt: trimmed } : undefined;
      const userMessage = {
        role: "user" as const,
        content: trimmed,
        attachments: images.length > 0 ? images : undefined,
        imageGeneration,
      };
      appendMessage(convId, userMessage);
      appendMessage(convId, { role: "assistant", content: "", imageGeneration });
      setAttachments([]);
      setAttachmentError(null);

      setIsStreaming(true);
      abortControllerRef.current = new AbortController();
      requestConversationRef.current = convId;

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

      const history: ReplayMessage[] = [
        ...(historyOverride ?? (previousResponseId ? [] : activeConversation?.messages ?? [])),
        userMessage,
      ];
      const requestController = abortControllerRef.current;

      const request = {
        model,
        mode: requestMode,
        prompt: trimmed,
        accessToken,
        signal: requestController.signal,
        messages: history,
        previousResponseId,
        mcpServers: selectedMCPServers,
        onResponseId: (id: string | null) => {
          if (responsesSessionGenerationRef.current === sessionGeneration) setResponsesSessionId(id);
        },
        onUpdate: (updates: AssistantMessageUpdate) => updateLastAssistantMessage(convId, updates),
      };
      try {
        await runChatRequest(request);
      } finally {
        if (abortControllerRef.current === requestController) {
          setIsStreaming(false);
          abortControllerRef.current = null;
          requestConversationRef.current = null;
        }
      }
    },
    [
      activeConversationId,
      activeConversation,
      selectedModel,
      selectedMCPServers,
      mode,
      attachments,
      accessToken,
      createConversation,
      appendMessage,
      updateLastAssistantMessage,
      isStreaming,
      responsesSessionId,
      setConversationIdInUrl,
    ],
  );

  const handleStop = useCallback(() => {
    abortControllerRef.current?.abort();
  }, []);

  const handleEditAndResend = useCallback(
    (messageId: string, newContent: string) => {
      if (!activeConversationId || isStreaming) return;
      if (abortControllerRef.current || readAttachmentsRef.current) return;
      // Compute the truncated history synchronously before the async state update lands,
      // so handleSend receives the correct pre-edit context rather than the stale closure value.
      const msgs = activeConversation?.messages ?? [];
      const idx = msgs.findIndex((m) => m.id === messageId);
      if (idx < 0) return;
      const original = msgs[idx];
      if (!original.imageGeneration && (mode !== "chat" || !selectedModel)) {
        setAttachmentError("Choose a chat model before resending this message.");
        return;
      }
      if (!newContent.trim() && !original.attachments?.length) return;
      const priorMessages = msgs.slice(0, idx);
      truncateFromMessage(activeConversationId, messageId);
      handleSend(newContent, priorMessages, original.attachments ?? [], original.imageGeneration?.model);
    },
    [activeConversationId, isStreaming, activeConversation, truncateFromMessage, handleSend, mode, selectedModel],
  );

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

  const inputBar = (inConversation: boolean) => {
    const preparing = isLoadingModels || isReadingAttachments;
    const submitDisabled = !hasMessageContent(inputText, attachments, mode) || !selectedModel || preparing;
    return (
      <div
        onDragOver={(event) => {
          if (mode === "chat" && event.dataTransfer.types.includes("Files")) event.preventDefault();
        }}
        onDrop={(event) => {
          if (mode === "chat" && event.dataTransfer.files.length) {
            event.preventDefault();
            void addAttachments(Array.from(event.dataTransfer.files));
          }
        }}
      >
        <MoyaiChatComposer
          inConversation={inConversation}
          value={inputText}
          onChange={setInputText}
          onSubmit={() => void handleSend(inputText)}
          onCancel={handleStop}
          placeholder={
            mode === "image"
              ? "Describe the image you want to create…"
              : inConversation
                ? "Send a message..."
                : "Describe a change, paste an error, or ask a question…"
          }
          isLoading={isStreaming}
          submitLabel={mode === "image" ? "Generate image" : "Send"}
          submitDisabled={submitDisabled}
          onPaste={(event) => {
            const images = Array.from(event.clipboardData.files).filter((file) => file.type.startsWith("image/"));
            if (mode === "chat" && images.length > 0) {
              event.preventDefault();
              void addAttachments(images);
            }
          }}
          attachmentPreview={
            <AttachmentPreview
              attachments={attachments}
              disabled={isStreaming}
              onRemove={(id) => setAttachments((current) => current.filter((item) => item.id !== id))}
            />
          }
          modelPicker={
            <ChatModelPicker
              models={models}
              selectedModel={selectedModel}
              loading={isLoadingModels}
              disabled={isStreaming || isReadingAttachments}
              onSelect={selectModel}
            />
          }
          tools={
            mode === "chat" ? (
              <>
                <input
                  ref={attachmentInputRef}
                  type="file"
                  accept={CHAT_ATTACHMENT_ACCEPT}
                  multiple
                  aria-label="Upload files"
                  className="sr-only"
                  disabled={isStreaming || isReadingAttachments}
                  onChange={handleAttachmentInput}
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-sm"
                  aria-label="Attach files"
                  title="Attach images, PDF, Word (.docx), Excel (.xlsx), text, or code files (up to 4 files, 10 MB each)"
                  disabled={isStreaming || isReadingAttachments || attachments.length >= MAX_CHAT_ATTACHMENTS}
                  onClick={() => attachmentInputRef.current?.click()}
                >
                  <Paperclip className="size-4" />
                </Button>
                <Popover open={mcpPopoverOpen} onOpenChange={setMcpPopoverOpen}>
                  <PopoverTrigger
                    render={
                      <Button
                        variant="ghost"
                        size="sm"
                        disabled={isStreaming}
                        aria-label="Connect tools"
                        className="gap-1.5 text-muted-foreground"
                      >
                        <Plus className="size-3.5" />
                        <span>Tools</span>
                        {selectedMCPServers.length > 0 && (
                          <span className="text-primary">{selectedMCPServers.length}</span>
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
              </>
            ) : (
              <span className="flex items-center gap-2 text-xs text-muted-foreground">
                <ImageIcon className="size-4" />
                Create an image from your description
              </span>
            )
          }
        />
        {isReadingAttachments && (
          <p role="status" className="mt-2 text-xs text-muted-foreground">
            Preparing files…
          </p>
        )}
        {attachments.some((attachment) => attachment.kind === "file") && (
          <p className="mt-2 text-xs text-muted-foreground">File support depends on the selected model and provider.</p>
        )}
        {attachmentError && (
          <p role="alert" className="mt-2 text-sm text-destructive">
            {attachmentError}
          </p>
        )}
        {!isLoadingModels && !selectedModel && (
          <p className="mt-2 text-sm text-muted-foreground">
            {mode === "image"
              ? "No image-generation models are available. Ask your admin to add one to Model Hub."
              : "No chat models are available for your account."}
          </p>
        )}
        <p className="moyai-chat-home-composer-hint">
          {mode === "image" ? "Describe an image · Enter to generate" : "Drop files or paste images · Enter to send"}
        </p>
      </div>
    );
  };

  return (
    <>
      {storageUnavailable && !storageBannerDismissed && (
        <div className="bg-warning/10 border-b border-warning/20 px-5 py-1.5 text-[13px] text-warning flex justify-between items-center">
          <span>Some chat history or attachments could not be saved or restored.</span>
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={() => setStorageBannerDismissed(true)}
            className="text-warning hover:bg-warning/15 hover:text-warning/80"
          >
            <X className="size-3.5" />
          </Button>
        </div>
      )}

      <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
        {showBlankState ? (
          <MoyaiChatHome
            imageMode={mode === "image"}
            draft={inputText}
            busy={isStreaming}
            conversations={conversations}
            onSelectConversation={(id) => {
              void setConversationIdInUrl(id);
            }}
            onPrompt={setInputText}
          >
            {inputBar(false)}
          </MoyaiChatHome>
        ) : (
          <div className="moyai-chat-conversation">
            <header className="moyai-chat-conversation-header">
              <h1>{activeConversation!.title}</h1>
            </header>
            <div className="moyai-chat-transcript-layout">
              <div ref={messagesScrollRef} className="moyai-chat-transcript" style={{ overflowAnchor: "none" }}>
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
                  className="absolute bottom-[100px] left-1/2 -translate-x-1/2 z-chrome rounded-full border bg-background/75 text-muted-foreground shadow-sm backdrop-blur-md hover:bg-background/95"
                  aria-label="Scroll to bottom"
                >
                  <ChevronDown className="h-3 w-3" />
                </Button>
              )}

              <div className="moyai-chat-composer-wrap">{inputBar(true)}</div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
