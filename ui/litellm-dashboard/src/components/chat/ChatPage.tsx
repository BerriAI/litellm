"use client";

import React, { useCallback, useEffect, useRef, useState, useLayoutEffect } from "react";
import { Select, Tooltip, Skeleton, Popover, message } from "antd";
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
} from "@ant-design/icons";
import { useRouter, useSearchParams } from "next/navigation";
import { useChatHistory } from "./useChatHistory";
import ConversationList from "./ConversationList";
import ChatMessages from "./ChatMessages";
import MCPConnectPicker from "./MCPConnectPicker";
import MCPAppsPanel from "./MCPAppsPanel";
import { fetchAvailableModels } from "../playground/llm_calls/fetch_models";
import { makeOpenAIChatCompletionRequest } from "../playground/llm_calls/chat_completion";
import { serverRootPath, getProxyBaseUrl } from "@/components/networking";

interface ChatPageProps {
  accessToken: string;
  userRole: string;
  userId: string;
  userEmail?: string;
}

const SUGGESTIONS = ["Write", "Learn", "Code", "Brainstorm"];

function getGreeting(): string {
  const h = new Date().getHours();
  if (h >= 5 && h < 12) return "Good morning";
  if (h >= 12 && h < 17) return "Good afternoon";
  return "Good evening";
}

const LOCALSTORAGE_MODEL_KEY = "litellm_chat_selected_model";

// Build the chat UI URL respecting server root path (e.g. /litellm/ui/chat)
function getChatUrl(id?: string): string {
  const root = serverRootPath && serverRootPath !== "/" ? serverRootPath.replace(/\/+$/, "") : "";
  return id ? `${root}/ui/chat?id=${id}` : `${root}/ui/chat`;
}

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
  const logoSrc = `${getProxyBaseUrl()}/get_image`;

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
    async (text: string, historyOverride?: Array<{ role: "user" | "assistant"; content: string }>) => {
      const trimmed = text.trim();
      if (!trimmed || !selectedModel || isStreaming) return;
      setInputText("");

      let convId = activeConversationId;
      if (!convId) {
        convId = createConversation(selectedModel);
        router.push(getChatUrl(convId));
      }

      appendMessage(convId, { role: "user", content: trimmed });
      appendMessage(convId, { role: "assistant", content: "" });

      setIsStreaming(true);
      abortControllerRef.current = new AbortController();

      const history = [
        ...(historyOverride ?? (activeConversation?.messages ?? [])
          .filter((m) => m.role === "user" || m.role === "assistant")
          .map((m) => ({
            role: m.role as "user" | "assistant",
            content: m.content,
          }))),
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
            content: accumulatedContent + " [stopped]",
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
      // Compute the truncated history synchronously before the async state update lands,
      // so handleSend receives the correct pre-edit context rather than the stale closure value.
      const msgs = activeConversation?.messages ?? [];
      const idx = msgs.findIndex((m) => m.id === messageId);
      const priorMessages = (idx === -1 ? msgs : msgs.slice(0, idx))
        .filter((m) => m.role === "user" || m.role === "assistant")
        .map((m) => ({ role: m.role as "user" | "assistant", content: m.content }));
      truncateAfterMessage(activeConversationId, messageId);
      handleSend(newContent, priorMessages);
    },
    [activeConversationId, isStreaming, activeConversation, truncateAfterMessage, handleSend],
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
          {sidebarNavItem(<EditOutlined />, "New chat", () => router.push(getChatUrl()))}
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
              onSelect={(id) => router.push(getChatUrl(id))}
              onDelete={deleteConversation}
              onNewChat={() => router.push(getChatUrl())}
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

          {/* Right: settings */}
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
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
                        background: "none", border: "1.5px solid #d1d5db", borderRadius: "50%",
                        width: 32, height: 32, cursor: "pointer", color: "#374151",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        flexShrink: 0,
                      }}>
                        <div style={{ width: 10, height: 10, background: "#374151", borderRadius: 2 }} />
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

            </div>
          ) : (
            /* ---- Active conversation ---- */
            <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column", maxWidth: 760, margin: "0 auto", width: "100%", padding: "0 24px", position: "relative" }}>
              <div ref={messagesScrollRef} style={{ flex: 1, minHeight: 0, overflow: "auto", paddingTop: 24, overflowAnchor: "none" }}>
                <ChatMessages
                  messages={activeConversation.messages}
                  isStreaming={isStreaming}
                  onEditMessage={handleEditAndResend}
                />
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
