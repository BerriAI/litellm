import React, { useEffect, useRef, useState } from "react";
import { Button, Select, Input, Spin } from "antd";
import ReactMarkdown from "react-markdown";
import { modelHubCall, usageAiChatStream, UsageAiToolCallEvent } from "../../networking";

const { TextArea } = Input;

interface ToolCallStep {
  tool_name: string;
  tool_label: string;
  arguments: Record<string, string>;
  status: "running" | "complete" | "error";
  error?: string;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  toolCalls?: ToolCallStep[];
}

interface UsageAIChatPanelProps {
  open: boolean;
  onClose: () => void;
  accessToken: string | null;
}

const TOOL_ICONS: Record<string, string> = {
  get_usage_data: "📊",
  get_team_usage_data: "👥",
  get_tag_usage_data: "🏷️",
};

const ToolCallDisplay: React.FC<{ step: ToolCallStep }> = ({ step }) => {
  const icon = TOOL_ICONS[step.tool_name] || "🔧";
  const args = step.arguments;
  const dateRange = args.start_date && args.end_date
    ? `${args.start_date} → ${args.end_date}`
    : "";
  const filter = args.team_ids || args.tags || args.user_id || "";

  return (
    <div className="flex items-start gap-2 px-3 py-2 rounded-lg bg-gray-100 border border-gray-200 text-xs">
      <span className="flex-shrink-0 mt-0.5">
        {step.status === "running" ? (
          <Spin size="small" />
        ) : step.status === "error" ? (
          <span className="text-red-500">✗</span>
        ) : (
          <span className="text-green-600">✓</span>
        )}
      </span>
      <div className="min-w-0">
        <div className="font-medium text-gray-700">
          {icon} {step.tool_label}
        </div>
        {dateRange && (
          <div className="text-gray-500 mt-0.5">{dateRange}</div>
        )}
        {filter && (
          <div className="text-gray-500 mt-0.5">Filter: {filter}</div>
        )}
        {step.status === "error" && step.error && (
          <div className="text-red-600 mt-0.5">{step.error}</div>
        )}
      </div>
    </div>
  );
};

const MarkdownContent: React.FC<{ content: string }> = ({ content }) => (
  <ReactMarkdown
    components={{
      p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
      strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
      ul: ({ children }) => <ul className="list-disc pl-4 mb-2 space-y-0.5">{children}</ul>,
      ol: ({ children }) => <ol className="list-decimal pl-4 mb-2 space-y-0.5">{children}</ol>,
      li: ({ children }) => <li>{children}</li>,
      h1: ({ children }) => <h4 className="font-semibold text-sm mt-2 mb-1">{children}</h4>,
      h2: ({ children }) => <h4 className="font-semibold text-sm mt-2 mb-1">{children}</h4>,
      h3: ({ children }) => <h4 className="font-semibold text-sm mt-2 mb-1">{children}</h4>,
      code: ({ children, className }) => {
        const isBlock = className?.includes("language-");
        return isBlock ? (
          <pre className="bg-gray-100 rounded p-2 my-1 overflow-x-auto text-xs">
            <code>{children}</code>
          </pre>
        ) : (
          <code className="px-1 py-0.5 rounded bg-gray-100 text-xs font-mono">{children}</code>
        );
      },
      table: ({ children }) => (
        <div className="overflow-x-auto my-2">
          <table className="text-xs border-collapse w-full">{children}</table>
        </div>
      ),
      th: ({ children }) => <th className="border border-gray-200 px-2 py-1 bg-gray-50 font-medium text-left">{children}</th>,
      td: ({ children }) => <td className="border border-gray-200 px-2 py-1">{children}</td>,
    }}
  >
    {content}
  </ReactMarkdown>
);

const UsageAIChatPanel: React.FC<UsageAIChatPanelProps> = ({
  open,
  onClose,
  accessToken,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | undefined>(undefined);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [activeToolCalls, setActiveToolCalls] = useState<ToolCallStep[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (open && availableModels.length === 0) {
      loadModels();
    }
  }, [open]);

  useEffect(() => {
    if (typeof messagesEndRef.current?.scrollIntoView === "function") {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, streamingContent, activeToolCalls, statusMessage]);

  const loadModels = async () => {
    if (!accessToken) return;
    setIsLoadingModels(true);
    try {
      const fetchedModels = await modelHubCall(accessToken);
      if (fetchedModels?.data?.length > 0) {
        const models = fetchedModels.data
          .map((item: any) => item.model_group as string)
          .sort();
        setAvailableModels(models);
      }
    } catch (error) {
      console.error("Failed to load models:", error);
    } finally {
      setIsLoadingModels(false);
    }
  };

  const handleSend = async () => {
    if (!accessToken || !inputText.trim() || isLoading) return;

    const userMessage: ChatMessage = { role: "user", content: inputText.trim() };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInputText("");
    setIsLoading(true);
    setStreamingContent("");
    setStatusMessage(null);
    setActiveToolCalls([]);

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    let accumulated = "";
    const toolCalls: ToolCallStep[] = [];

    try {
      await usageAiChatStream(
        accessToken,
        updatedMessages.slice(-20).map((m) => ({ role: m.role, content: m.content })),
        selectedModel || "",
        (content: string) => {
          setStatusMessage(null);
          accumulated += content;
          setStreamingContent(accumulated);
        },
        () => {
          setStatusMessage(null);
          setActiveToolCalls([]);
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: accumulated, toolCalls: toolCalls.length > 0 ? [...toolCalls] : undefined },
          ]);
          setStreamingContent("");
        },
        (errorMsg: string) => {
          setStatusMessage(null);
          setActiveToolCalls([]);
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: `Error: ${errorMsg}` },
          ]);
          setStreamingContent("");
        },
        (status: string) => {
          setStatusMessage(status);
        },
        (event: UsageAiToolCallEvent) => {
          const idx = toolCalls.findIndex((tc) => tc.tool_name === event.tool_name);
          if (idx >= 0) {
            toolCalls[idx] = { ...event };
          } else {
            toolCalls.push({ ...event });
          }
          setActiveToolCalls([...toolCalls]);
        },
        abortController.signal,
      );
    } catch (error: any) {
      if (error?.name === "AbortError" || abortController.signal.aborted) {
        return;
      }
      const errorMsg = error?.message || "Failed to get response. Please try again.";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${errorMsg}` },
      ]);
      setStreamingContent("");
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClose = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    onClose();
  };

  const handleClear = () => {
    setMessages([]);
    setStreamingContent("");
    setActiveToolCalls([]);
    setStatusMessage(null);
  };

  return (
    <div
      data-testid="usage-ai-chat-panel"
      className={`fixed top-0 right-0 h-full bg-white border-l border-gray-200 shadow-2xl z-50 flex flex-col transition-transform duration-300 ease-in-out ${
        open ? "translate-x-0" : "translate-x-full"
      }`}
      style={{ width: 420 }}
    >
      {/* Header */}
      <div className="px-5 pt-5 pb-3 border-b border-gray-100 flex-shrink-0">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-blue-600" viewBox="0 0 16 16" fill="currentColor">
              <path d="M8 1l1.5 3.5L13 6l-3.5 1.5L8 11 6.5 7.5 3 6l3.5-1.5L8 1zm4 7l.75 1.75L14.5 10.5l-1.75.75L12 13l-.75-1.75L9.5 10.5l1.75-.75L12 8zM4 9l.75 1.75L6.5 11.5l-1.75.75L4 14l-.75-1.75L1.5 11.5l1.75-.75L4 9z" />
            </svg>
            <h3 className="text-base font-semibold text-gray-900">Ask AI</h3>
          </div>
          <button
            onClick={handleClose}
            className="text-gray-400 hover:text-gray-600 transition-colors p-1 rounded-md hover:bg-gray-100"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        <p className="text-xs text-gray-500">
          Ask about your spend, models, keys, and trends
        </p>
      </div>

      {/* Model selector */}
      <div className="px-5 py-3 border-b border-gray-100 flex-shrink-0">
        <Select
          placeholder="Select a model (optional, defaults to gpt-4o-mini)"
          value={selectedModel}
          onChange={(value) => setSelectedModel(value)}
          loading={isLoadingModels}
          showSearch
          allowClear
          size="small"
          className="w-full"
          options={availableModels.map((m) => ({ label: m, value: m }))}
          filterOption={(input, option) =>
            (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
          }
        />
      </div>

      {/* Chat messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-gray-50">
        {messages.length === 0 && !streamingContent && !isLoading && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <svg className="w-8 h-8 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
            </svg>
            <p className="text-sm font-medium">Ask a question about your usage</p>
            <p className="text-xs mt-1">e.g. &quot;Which model costs me the most?&quot;</p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx}>
            {msg.role === "user" ? (
              <div className="flex justify-end">
                <div className="max-w-[88%] rounded-xl px-3.5 py-2 text-sm leading-relaxed bg-blue-600 text-white">
                  {msg.content}
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                {/* Tool calls for this message */}
                {msg.toolCalls && msg.toolCalls.length > 0 && (
                  <div className="space-y-1.5">
                    {msg.toolCalls.map((tc, tcIdx) => (
                      <ToolCallDisplay key={tcIdx} step={tc} />
                    ))}
                  </div>
                )}
                {/* Response */}
                <div className="max-w-[95%] rounded-xl px-3.5 py-2.5 text-sm leading-relaxed bg-white border border-gray-200 text-gray-800">
                  <MarkdownContent content={msg.content} />
                </div>
              </div>
            )}
          </div>
        ))}

        {/* Active tool calls (in-progress) */}
        {isLoading && activeToolCalls.length > 0 && (
          <div className="space-y-1.5">
            {activeToolCalls.map((tc, idx) => (
              <ToolCallDisplay key={idx} step={tc} />
            ))}
          </div>
        )}

        {/* Status / spinner */}
        {isLoading && !streamingContent && (
          <div className="flex items-center gap-2 px-3 py-2 text-xs text-gray-500">
            <Spin size="small" />
            <span className="italic">{statusMessage || "Thinking..."}</span>
          </div>
        )}

        {/* Streaming response */}
        {streamingContent && (
          <div className="max-w-[95%] rounded-xl px-3.5 py-2.5 text-sm leading-relaxed bg-white border border-gray-200 text-gray-800">
            <MarkdownContent content={streamingContent} />
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="px-4 py-3 border-t border-gray-200 bg-white flex-shrink-0">
        <div className="flex gap-2">
          <TextArea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your usage..."
            autoSize={{ minRows: 1, maxRows: 3 }}
            className="flex-1"
            disabled={isLoading}
          />
          <Button
            type="primary"
            onClick={handleSend}
            disabled={!inputText.trim() || isLoading}
            loading={isLoading}
          >
            Send
          </Button>
        </div>
        <div className="flex justify-between items-center mt-2">
          <button
            onClick={handleClear}
            className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
            disabled={messages.length === 0}
          >
            Clear chat
          </button>
          <span className="text-xs text-gray-400">
            Enter to send
          </span>
        </div>
      </div>
    </div>
  );
};

export default UsageAIChatPanel;
