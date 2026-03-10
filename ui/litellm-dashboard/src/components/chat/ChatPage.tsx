"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { Select, Tooltip, Skeleton, Popover, message } from "antd";
import {
  SettingOutlined,
  PlusOutlined,
  BorderOutlined,
  EditOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  SearchOutlined,
  MessageOutlined,
  AppstoreOutlined,
  ArrowLeftOutlined,
} from "@ant-design/icons";
import { useRouter, useSearchParams } from "next/navigation";
import { useChatHistory } from "./useChatHistory";
import ConversationList from "./ConversationList";
import ChatMessages from "./ChatMessages";
import MCPConnectPicker from "./MCPConnectPicker";
import MCPAppsPanel from "./MCPAppsPanel";
import { fetchAvailableModels } from "../playground/llm_calls/fetch_models";
import { makeOpenAIChatCompletionRequest } from "../playground/llm_calls/chat_completion";
import { serverRootPath } from "@/components/networking";

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

// Build the dashboard root URL
function getDashboardUrl(): string {
  const base = process.env.NEXT_PUBLIC_BASE_URL ?? "";
  const trimmed = base.replace(/^\/+|\/+$/g, "");
  const uiPath = trimmed ? `/${trimmed}/` : "/";
  if (serverRootPath && serverRootPath !== "/") {
    const cleanRoot = serverRootPath.replace(/\/+$/, "");
    const cleanUi = uiPath.replace(/^\/+/, "");
    return `${cleanRoot}/${cleanUi}`;
  }
  return uiPath;
}

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
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarView, setSidebarView] = useState<"chats" | "apps">("chats");
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
    truncateAfterMessage,
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
        const saved = localStorage.getItem(LOCALSTORAGE_MODEL_KEY);
        if (saved && names.includes(saved)) {
          setSelectedModel(saved);
        } else if (names.length > 0) {
          setSelectedModel(names[0]);
          localStorage.setItem(LOCALSTORAGE_MODEL_KEY, names[0]);
        }
      })
      .catch(() => message.error("Could not load models"))
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

      let accumulatedContent = "";
      let accumulatedReasoning = "";

      try {
        await makeOpenAIChatCompletionRequest(
          history,
          (chunk: string) => {
            accumulatedContent += chunk;
            updateLastAssistantMessage(convId!, { content: accumulatedContent });
          },
          selectedModel,
          accessToken,
          undefined,
          abortControllerRef.current.signal,
          (rc: string) => {
            accumulatedReasoning += rc;
            updateLastAssistantMessage(convId!, { reasoningContent: accumulatedReasoning });
          },
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

  const handleEditAndResend = useCallback(
    (messageId: string, newContent: string) => {
      if (!activeConversationId || isStreaming) return;
      // Remove the edited message and everything after it
      truncateAfterMessage(activeConversationId, messageId);
      // Re-send with the new content (handleSend appends user msg + starts completion)
      handleSend(newContent);
    },
    [activeConversationId, isStreaming, truncateAfterMessage, handleSend],
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

  const showBlankState = !activeConversation || activeConversation.messages.length === 0;
  const displayName = userEmail?.split("@")[0] ?? userId ?? "";
  const greeting = displayName ? `${getGreeting()}, ${displayName}` : getGreeting();
  const dashboardUrl = getDashboardUrl();

  // ---- Sidebar nav item renderer (inline, not a function-in-function) ----
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
                src="/assets/logos/litellm_logo.jpg"
                alt="LiteLLM"
                width={26}
                height={26}
                style={{ borderRadius: 6, objectFit: "cover", flexShrink: 0 }}
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
          {sidebarNavItem(<EditOutlined />, "New chat", () => router.push("/chat"))}
          {sidebarNavItem(<SearchOutlined />, "Search chats", () => {}, false, "⌘K")}
        </div>

        <div style={{ height: 1, background: "#e5e7eb", margin: "4px 8px", flexShrink: 0 }} />

        {/* Chats / Apps tabs */}
        <div style={{ padding: "4px 8px", flexShrink: 0 }}>
          {sidebarNavItem(<MessageOutlined />, "Chats", () => setSidebarView("chats"), sidebarView === "chats")}
          {sidebarNavItem(<AppstoreOutlined />, "Apps", () => setSidebarView("apps"), sidebarView === "apps")}
        </div>

        <div style={{ height: 1, background: "#e5e7eb", margin: "4px 8px", flexShrink: 0 }} />

        {/* Sidebar content — only conversation list, only when in chats view and expanded */}
        {!sidebarCollapsed && sidebarView === "chats" && (
          <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
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
      </div>

      {/* ===== MAIN AREA ===== */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", minWidth: 0 }}>

        {/* Top bar — clean, minimal like ChatGPT */}
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
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {isLoadingModels ? (
              <Skeleton.Input active style={{ width: 160, height: 28 }} />
            ) : (
              <Select
                value={selectedModel || undefined}
                onChange={handleModelChange}
                showSearch
                placeholder="Select model"
                style={{ width: 220 }}
                size="middle"
                variant="borderless"
                options={models.map((m) => ({
                  value: m,
                  label: m.length > 35 ? m.slice(0, 35) + "…" : m,
                }))}
              />
            )}
          </div>

          {/* Right: back to dashboard + settings */}
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <Tooltip title="Back to Dashboard">
              <a
                href={dashboardUrl}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  padding: "5px 12px",
                  borderRadius: 7,
                  border: "1px solid #e5e7eb",
                  color: "#374151",
                  fontSize: 13,
                  fontWeight: 500,
                  textDecoration: "none",
                  background: "#fff",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLAnchorElement).style.background = "#f9fafb";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLAnchorElement).style.background = "#fff";
                }}
              >
                <ArrowLeftOutlined style={{ fontSize: 12 }} />
                Dashboard
              </a>
            </Tooltip>
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
        <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column", background: "#fff" }}>
          {/* ---- Apps page view ---- */}
          {sidebarView === "apps" ? (
            <div style={{ flex: 1, maxWidth: 800, margin: "0 auto", width: "100%", padding: "32px 24px" }}>
              <div style={{ marginBottom: 24 }}>
                <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, color: "#111827" }}>MCP Servers</h1>
                <p style={{ margin: "6px 0 0", fontSize: 14, color: "#6b7280" }}>
                  Connect tools to your chat. Toggled servers are active in every new message.
                </p>
              </div>
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
                {greeting}
              </h1>

              {/* Input card */}
              <div style={{
                width: "100%",
                maxWidth: 680,
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
                  placeholder="How can I help you today?"
                  style={{
                    width: "100%",
                    minHeight: 80,
                    padding: "20px 20px 8px",
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
                  padding: "8px 12px 12px",
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
                    <Tooltip title="Attach tools">
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
                    </Tooltip>
                  </Popover>

                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ fontSize: 12, color: "#9ca3af", maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {selectedModel || "No model"}
                    </span>
                    {isStreaming ? (
                      <button onClick={handleStop} style={{
                        background: "#111827", border: "none", borderRadius: 7,
                        padding: "7px 10px", cursor: "pointer", color: "#fff",
                        display: "flex", alignItems: "center",
                      }}>
                        <BorderOutlined style={{ fontSize: 14 }} />
                      </button>
                    ) : (
                      <button
                        onClick={() => handleSend(inputText)}
                        disabled={!inputText.trim() || isLoadingModels || !selectedModel}
                        style={{
                          background: inputText.trim() && selectedModel ? "#1677ff" : "#f3f4f6",
                          border: "none", borderRadius: 7,
                          padding: "7px 16px", cursor: inputText.trim() && selectedModel ? "pointer" : "not-allowed",
                          color: inputText.trim() && selectedModel ? "#fff" : "#9ca3af",
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

              {/* Suggestion chips */}
              <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap", justifyContent: "center" }}>
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s.label}
                    onClick={() => setInputText(s.label + ": ")}
                    style={{
                      background: "#f9fafb",
                      border: "1px solid #e5e7eb",
                      borderRadius: 20,
                      padding: "7px 16px",
                      fontSize: 14,
                      color: "#374151",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                    }}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLButtonElement).style.background = "#f3f4f6";
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLButtonElement).style.background = "#f9fafb";
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
                  onEditMessage={handleEditAndResend}
                />
              </div>

              {/* Input bar (in conversation) */}
              <div style={{ padding: "12px 0 24px" }}>
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
                    placeholder="Send a message..."
                    style={{
                      width: "100%",
                      minHeight: 52,
                      padding: "16px 20px 8px",
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
                    padding: "4px 12px 10px",
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
                      <span style={{ fontSize: 12, color: "#9ca3af" }}>
                        {selectedMCPServers.length > 0 ? `${selectedMCPServers.length} tool${selectedMCPServers.length > 1 ? "s" : ""} connected` : ""}
                      </span>
                      {isStreaming ? (
                        <button onClick={handleStop} style={{
                          background: "#111827", border: "none", borderRadius: 7,
                          padding: "7px 10px", cursor: "pointer", color: "#fff",
                          display: "flex", alignItems: "center",
                        }}>
                          <BorderOutlined style={{ fontSize: 14 }} />
                        </button>
                      ) : (
                        <button
                          onClick={() => handleSend(inputText)}
                          disabled={!inputText.trim() || isLoadingModels || !selectedModel}
                          style={{
                            background: inputText.trim() && selectedModel ? "#1677ff" : "#f3f4f6",
                            border: "none", borderRadius: 7,
                            padding: "7px 16px", cursor: inputText.trim() && selectedModel ? "pointer" : "not-allowed",
                            color: inputText.trim() && selectedModel ? "#fff" : "#9ca3af",
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
