"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Select, Tooltip, Skeleton, Popover, message } from "antd";
import { SettingOutlined, PlusOutlined, BorderOutlined } from "@ant-design/icons";
import { useRouter, useSearchParams } from "next/navigation";
import { useChatHistory } from "./useChatHistory";
import ConversationList from "./ConversationList";
import ChatMessages from "./ChatMessages";
import MCPConnectPicker from "./MCPConnectPicker";
import { fetchAvailableModels } from "../playground/llm_calls/fetch_models";
import { makeOpenAIChatCompletionRequest } from "../playground/llm_calls/chat_completion";

interface ChatPageProps {
  accessToken: string;
  userRole: string;
  userId: string;
  userEmail?: string;
}

const SUGGESTIONS = [
  { label: "Write", icon: "✏️" },
  { label: "Learn", icon: "🎓" },
  { label: "Code", icon: "</>" },
  { label: "Brainstorm", icon: "💡" },
];

function getGreeting(): string {
  const h = new Date().getHours();
  if (h >= 5 && h < 12) return "Good morning";
  if (h >= 12 && h < 17) return "Good afternoon";
  return "Good evening";
}

const LOCALSTORAGE_MODEL_KEY = "litellm_chat_selected_model";

const ChatPage: React.FC<ChatPageProps> = ({ accessToken, userRole, userId, userEmail }) => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const activeConversationId = searchParams.get("id");

  const [selectedModel, setSelectedModel] = useState<string>("");
  const [models, setModels] = useState<string[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [selectedMCPServers, setSelectedMCPServers] = useState<string[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [inputText, setInputText] = useState("");
  const [mcpPopoverOpen, setMcpPopoverOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [storageBannerDismissed, setStorageBannerDismissed] = useState(false);

  const abortControllerRef = useRef<AbortController | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const {
    conversations,
    activeConversation,
    storageUnavailable,
    staleId,
    createConversation,
    appendMessage,
    updateLastAssistantMessage,
    deleteConversation,
    renameConversation,
  } = useChatHistory(activeConversationId);

  // Load models
  useEffect(() => {
    if (!accessToken) return;
    setIsLoadingModels(true);
    fetchAvailableModels(accessToken)
      .then((data) => {
        const names = (data || []).map((m: { model_name?: string }) => m.model_name ?? "").filter(Boolean);
        setModels(names);
        const saved = localStorage.getItem(LOCALSTORAGE_MODEL_KEY);
        if (saved && names.includes(saved)) {
          setSelectedModel(saved);
        } else if (names.length > 0) {
          setSelectedModel(names[0]);
          localStorage.setItem(LOCALSTORAGE_MODEL_KEY, names[0]);
        }
      })
      .catch(() => {
        message.error("Could not load models");
      })
      .finally(() => setIsLoadingModels(false));
  }, [accessToken]);

  useEffect(() => {
    if (staleId) router.replace("/chat");
  }, [staleId, router]);

  const handleModelChange = (val: string) => {
    setSelectedModel(val);
    localStorage.setItem(LOCALSTORAGE_MODEL_KEY, val);
  };

  const handleSend = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || !selectedModel || isStreaming) return;
      setInputText("");

      let convId = activeConversationId;
      if (!convId) {
        convId = createConversation(selectedModel);
        router.push(`/chat?id=${convId}`);
      }

      appendMessage(convId, { role: "user", content: trimmed });
      appendMessage(convId, { role: "assistant", content: "" });

      setIsStreaming(true);
      abortControllerRef.current = new AbortController();

      const history = [
        ...(activeConversation?.messages ?? []).map((m) => ({
          role: m.role as "user" | "assistant",
          content: m.content,
        })),
        { role: "user" as const, content: trimmed },
      ];

      try {
        await makeOpenAIChatCompletionRequest(
          history,
          (chunk: string) => updateLastAssistantMessage(convId!, { content: chunk }),
          selectedModel,
          accessToken,
          undefined,
          abortControllerRef.current.signal,
          (rc: string) => updateLastAssistantMessage(convId!, { reasoningContent: rc }),
          undefined, undefined, undefined, undefined, undefined, undefined,
          selectedMCPServers.length > 0 ? selectedMCPServers : undefined,
        );
      } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") {
          updateLastAssistantMessage(convId!, {
            content: (activeConversation?.messages.at(-1)?.content ?? "") + " [stopped]",
          });
        } else {
          updateLastAssistantMessage(convId!, {
            content: "[Something went wrong. The partial response has been saved.]",
          });
        }
      } finally {
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    },
    [activeConversationId, activeConversation, selectedModel, selectedMCPServers, accessToken,
      createConversation, appendMessage, updateLastAssistantMessage, router, isStreaming],
  );

  const handleStop = useCallback(() => {
    abortControllerRef.current?.abort();
  }, []);

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

  const showBlankState = !activeConversation || activeConversation.messages.length === 0;
  const displayName = userEmail?.split("@")[0] ?? userId ?? "there";
  const greeting = `${getGreeting()}, ${displayName}`;

  return (
    <div style={{
      display: "flex",
      height: "100vh",
      width: "100vw",
      background: "#ffffff",
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      overflow: "hidden",
    }}>

      {/* Conversation sidebar — slides in */}
      {sidebarOpen && (
        <div style={{
          width: 260,
          flexShrink: 0,
          background: "#fafafa",
          borderRight: "1px solid #f0f0f0",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}>
          <ConversationList
            conversations={conversations}
            activeConversationId={activeConversationId}
            onSelect={(id) => router.push(`/chat?id=${id}`)}
            onDelete={deleteConversation}
            onNewChat={() => router.push("/chat")}
            onRename={renameConversation}
          />
        </div>
      )}

      {/* Main area */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

        {/* Top bar */}
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px 20px",
          flexShrink: 0,
          borderBottom: "1px solid #f0f0f0",
          background: "#fff",
        }}>
          {/* Left: sidebar toggle + model selector */}
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <button
              onClick={() => setSidebarOpen((v) => !v)}
              style={{
                background: "none", border: "none", cursor: "pointer",
                padding: 6, borderRadius: 6, color: "#595959",
                fontSize: 18, lineHeight: 1,
              }}
              title="Toggle chat history"
            >
              ☰
            </button>
            {isLoadingModels ? (
              <Skeleton.Input active style={{ width: 160, height: 32 }} />
            ) : (
              <Select
                value={selectedModel || undefined}
                onChange={handleModelChange}
                showSearch
                placeholder="Select model"
                style={{ width: 220 }}
                size="middle"
                variant="filled"
                options={models.map((m) => ({
                  value: m,
                  label: m.length > 35 ? m.slice(0, 35) + "…" : m,
                }))}
              />
            )}
          </div>

          {/* Center: LiteLLM logo */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <img
              src="/assets/logos/litellm_logo.jpg"
              alt="LiteLLM"
              width={28}
              height={28}
              style={{ borderRadius: 6, objectFit: "cover" }}
            />
            <span style={{ fontWeight: 600, fontSize: 15, color: "#1f2937", letterSpacing: "-0.01em" }}>
              LiteLLM Chat
            </span>
          </div>

          {/* Right: settings */}
          <Tooltip title="Settings">
            <button style={{
              background: "none", border: "none", cursor: "pointer",
              padding: 6, borderRadius: 6, color: "#595959", fontSize: 18,
            }}>
              <SettingOutlined />
            </button>
          </Tooltip>
        </div>

        {storageUnavailable && !storageBannerDismissed && (
          <div style={{
            background: "#fffbe6", borderBottom: "1px solid #ffe58f",
            padding: "8px 20px", fontSize: 13, color: "#874d00",
            display: "flex", justifyContent: "space-between", alignItems: "center",
          }}>
            <span>Chat history won&apos;t be saved in this browser session.</span>
            <button onClick={() => setStorageBannerDismissed(true)}
              style={{ background: "none", border: "none", cursor: "pointer", fontSize: 16, color: "#92400e" }}>
              ×
            </button>
          </div>
        )}

        {/* Content area */}
        <div style={{ flex: 1, overflow: "auto", display: "flex", flexDirection: "column", background: "#f9fafb" }}>
          {showBlankState ? (
            /* ---- Blank state ---- */
            <div style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              padding: "0 24px 60px",
            }}>
              {/* Greeting */}
              <div style={{
                display: "flex",
                alignItems: "center",
                gap: 14,
                marginBottom: 40,
              }}>
                <img
                  src="/assets/logos/litellm_logo.jpg"
                  alt="LiteLLM"
                  width={40}
                  height={40}
                  style={{ borderRadius: 8, objectFit: "cover" }}
                />
                <h1 style={{
                  margin: 0,
                  fontSize: 32,
                  fontWeight: 600,
                  color: "#1f2937",
                  fontFamily: "inherit",
                  letterSpacing: "-0.01em",
                }}>
                  {greeting}
                </h1>
              </div>

              {/* Input card */}
              <div style={{
                width: "100%",
                maxWidth: 680,
                background: "#fff",
                borderRadius: 12,
                border: "1px solid #e8e8e8",
                boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
                overflow: "hidden",
              }}>
                <textarea
                  ref={textareaRef}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="How can I help you today?"
                  style={{
                    width: "100%",
                    minHeight: 80,
                    padding: "20px 20px 8px",
                    border: "none",
                    outline: "none",
                    resize: "none",
                    fontSize: 15,
                    color: "#1f2937",
                    background: "transparent",
                    fontFamily: "inherit",
                    boxSizing: "border-box",
                  }}
                />
                {/* Card footer */}
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "8px 12px 12px",
                  borderTop: "1px solid #f5f5f5",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
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
                        background: "none", border: "1px solid #d9d9d9",
                        borderRadius: 6, padding: "5px 10px",
                        cursor: "pointer", fontSize: 16, color: "#595959",
                        display: "flex", alignItems: "center",
                      }}>
                        <PlusOutlined />
                      </button>
                    </Popover>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: 12, color: "#8c8c8c", maxWidth: 140, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {selectedModel || "No model"}
                    </span>
                    {isStreaming ? (
                      <button
                        onClick={handleStop}
                        style={{
                          background: "#ff4d4f", border: "none", borderRadius: 6,
                          padding: "6px 8px", cursor: "pointer", color: "#fff",
                          display: "flex", alignItems: "center",
                        }}
                      >
                        <BorderOutlined />
                      </button>
                    ) : (
                      <button
                        onClick={() => handleSend(inputText)}
                        disabled={!inputText.trim() || isLoadingModels || !selectedModel}
                        style={{
                          background: inputText.trim() && selectedModel ? "#1677ff" : "#f0f0f0",
                          border: "none", borderRadius: 6,
                          padding: "6px 14px", cursor: inputText.trim() && selectedModel ? "pointer" : "not-allowed",
                          color: inputText.trim() && selectedModel ? "#fff" : "#bfbfbf",
                          display: "flex", alignItems: "center",
                          transition: "background 0.15s",
                          fontSize: 14, fontWeight: 500,
                        }}
                      >
                        Send
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Suggestion chips */}
              <div style={{ display: "flex", gap: 10, marginTop: 16, flexWrap: "wrap", justifyContent: "center" }}>
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s.label}
                    onClick={() => setInputText(s.label + ": ")}
                    style={{
                      background: "#fff",
                      border: "1px solid #e8e8e8",
                      borderRadius: 8,
                      padding: "7px 16px",
                      fontSize: 14,
                      color: "#595959",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                    }}
                  >
                    <span>{s.icon}</span> {s.label}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* ---- Active conversation ---- */
            <div style={{ flex: 1, display: "flex", flexDirection: "column", maxWidth: 760, margin: "0 auto", width: "100%", padding: "0 24px" }}>
              <div style={{ flex: 1, overflow: "auto", paddingTop: 24 }}>
                <ChatMessages
                  messages={activeConversation.messages}
                  isStreaming={isStreaming}
                />
              </div>

              {/* Input bar (in conversation) */}
              <div style={{ padding: "12px 0 24px" }}>
                <div style={{
                  background: "#fff",
                  borderRadius: 12,
                  border: "1px solid #e8e8e8",
                  boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
                  overflow: "hidden",
                }}>
                  <textarea
                    ref={textareaRef}
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Reply..."
                    style={{
                      width: "100%",
                      minHeight: 56,
                      padding: "16px 20px 8px",
                      border: "none",
                      outline: "none",
                      resize: "none",
                      fontSize: 15,
                      color: "#1f2937",
                      background: "transparent",
                      fontFamily: "inherit",
                      boxSizing: "border-box",
                    }}
                  />
                  <div style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "4px 12px 10px",
                    borderTop: "1px solid #f5f5f5",
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
                        background: "none", border: "1px solid #d9d9d9",
                        borderRadius: 6, padding: "5px 10px",
                        cursor: "pointer", fontSize: 14, color: "#595959",
                        display: "flex", alignItems: "center",
                      }}>
                        <PlusOutlined />
                      </button>
                    </Popover>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: 12, color: "#8c8c8c" }}>
                        {selectedMCPServers.length > 0 ? `MCP (${selectedMCPServers.length})` : ""}
                      </span>
                      {isStreaming ? (
                        <button onClick={handleStop} style={{
                          background: "#ff4d4f", border: "none", borderRadius: 6,
                          padding: "6px 8px", cursor: "pointer", color: "#fff",
                          display: "flex", alignItems: "center",
                        }}>
                          <BorderOutlined />
                        </button>
                      ) : (
                        <button
                          onClick={() => handleSend(inputText)}
                          disabled={!inputText.trim() || isLoadingModels || !selectedModel}
                          style={{
                            background: inputText.trim() && selectedModel ? "#1677ff" : "#f0f0f0",
                            border: "none", borderRadius: 6,
                            padding: "6px 14px", cursor: inputText.trim() && selectedModel ? "pointer" : "not-allowed",
                            color: inputText.trim() && selectedModel ? "#fff" : "#bfbfbf",
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
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatPage;
