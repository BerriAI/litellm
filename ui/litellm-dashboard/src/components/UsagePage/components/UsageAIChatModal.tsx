import React, { useEffect, useMemo, useRef, useState } from "react";
import { Modal, Select, Input, Spin } from "antd";
import { Button } from "@tremor/react";
import { getProxyBaseUrl, modelHubCall } from "../../networking";
import { DailyData } from "../types";
import openai from "openai";

const { TextArea } = Input;

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface UsageAIChatModalProps {
  visible: boolean;
  onCancel: () => void;
  accessToken: string | null;
  userSpendData: {
    results: DailyData[];
    metadata: any;
  };
  dateRange: {
    from?: Date;
    to?: Date;
  };
}

function buildUsageSummary(
  userSpendData: UsageAIChatModalProps["userSpendData"],
  dateRange: UsageAIChatModalProps["dateRange"]
): string {
  const meta = userSpendData.metadata || {};
  const results = userSpendData.results || [];

  const fromStr = dateRange.from?.toLocaleDateString() ?? "N/A";
  const toStr = dateRange.to?.toLocaleDateString() ?? "N/A";

  const modelSpend: Record<string, { spend: number; requests: number; tokens: number }> = {};
  const providerSpend: Record<string, { spend: number; requests: number }> = {};
  const keySpend: Record<string, { spend: number; alias: string | null }> = {};

  for (const day of results) {
    for (const [model, metrics] of Object.entries(day.breakdown.models || {})) {
      if (!modelSpend[model]) modelSpend[model] = { spend: 0, requests: 0, tokens: 0 };
      modelSpend[model].spend += metrics.metrics.spend;
      modelSpend[model].requests += metrics.metrics.api_requests;
      modelSpend[model].tokens += metrics.metrics.total_tokens;
    }
    for (const [provider, metrics] of Object.entries(day.breakdown.providers || {})) {
      if (!providerSpend[provider]) providerSpend[provider] = { spend: 0, requests: 0 };
      providerSpend[provider].spend += metrics.metrics.spend;
      providerSpend[provider].requests += metrics.metrics.api_requests;
    }
    for (const [key, metrics] of Object.entries(day.breakdown.api_keys || {})) {
      if (!keySpend[key]) keySpend[key] = { spend: 0, alias: metrics.metadata.key_alias };
      keySpend[key].spend += metrics.metrics.spend;
    }
  }

  const topModels = Object.entries(modelSpend)
    .sort((a, b) => b[1].spend - a[1].spend)
    .slice(0, 10)
    .map(([name, d]) => `  - ${name}: $${d.spend.toFixed(4)} (${d.requests} requests, ${d.tokens} tokens)`)
    .join("\n");

  const topProviders = Object.entries(providerSpend)
    .sort((a, b) => b[1].spend - a[1].spend)
    .slice(0, 10)
    .map(([name, d]) => `  - ${name}: $${d.spend.toFixed(4)} (${d.requests} requests)`)
    .join("\n");

  const topKeys = Object.entries(keySpend)
    .sort((a, b) => b[1].spend - a[1].spend)
    .slice(0, 10)
    .map(([key, d]) => `  - ${d.alias || key}: $${d.spend.toFixed(4)}`)
    .join("\n");

  const dailySummary = results
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
    .map((d) => `  - ${d.date}: $${d.metrics.spend.toFixed(4)} (${d.metrics.api_requests} requests)`)
    .join("\n");

  return `Date Range: ${fromStr} to ${toStr}
Total Spend: $${(meta.total_spend || 0).toFixed(4)}
Total Requests: ${meta.total_api_requests || 0}
Successful Requests: ${meta.total_successful_requests || 0}
Failed Requests: ${meta.total_failed_requests || 0}
Total Tokens: ${meta.total_tokens || 0}

Top Models by Spend:
${topModels || "  (no data)"}

Top Providers by Spend:
${topProviders || "  (no data)"}

Top API Keys by Spend:
${topKeys || "  (no data)"}

Daily Spend:
${dailySummary || "  (no data)"}`;
}

const SYSTEM_PROMPT = `You are an AI assistant that helps users understand their LLM API usage data. You are embedded in the LiteLLM Usage dashboard.

You have access to the user's current usage data which is provided below. Use it to answer questions about their spending, model usage, API key activity, provider costs, request volumes, and trends.

Be concise and helpful. Use specific numbers from the data. When discussing costs, format them as dollar amounts. If the user asks about something not available in the data, let them know.`;

const UsageAIChatModal: React.FC<UsageAIChatModalProps> = ({
  visible,
  onCancel,
  accessToken,
  userSpendData,
  dateRange,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string | undefined>(undefined);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (visible && availableModels.length === 0) {
      loadModels();
    }
  }, [visible]);

  useEffect(() => {
    if (typeof messagesEndRef.current?.scrollIntoView === "function") {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, streamingContent]);

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

  const usageSummary = useMemo(
    () => buildUsageSummary(userSpendData, dateRange),
    [userSpendData, dateRange]
  );

  const handleSend = async () => {
    if (!accessToken || !inputText.trim() || !selectedModel || isLoading) return;

    const userMessage: ChatMessage = { role: "user", content: inputText.trim() };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInputText("");
    setIsLoading(true);
    setStreamingContent("");

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const client = new openai.OpenAI({
        apiKey: accessToken,
        baseURL: proxyBaseUrl,
        dangerouslyAllowBrowser: true,
      });

      const chatHistory = [
        {
          role: "system" as const,
          content: `${SYSTEM_PROMPT}\n\nCurrent Usage Data:\n${usageSummary}`,
        },
        ...updatedMessages.map((m) => ({
          role: m.role as "user" | "assistant",
          content: m.content,
        })),
      ];

      const response = await client.chat.completions.create(
        {
          model: selectedModel,
          stream: true,
          messages: chatHistory,
        },
        { signal: abortController.signal }
      );

      let fullContent = "";
      for await (const chunk of response) {
        if (chunk.choices[0]?.delta?.content) {
          fullContent += chunk.choices[0].delta.content;
          setStreamingContent(fullContent);
        }
      }

      setMessages((prev) => [...prev, { role: "assistant", content: fullContent }]);
      setStreamingContent("");
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

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    onCancel();
  };

  const handleClear = () => {
    setMessages([]);
    setStreamingContent("");
  };

  return (
    <Modal
      title={null}
      open={visible}
      onCancel={handleCancel}
      width={720}
      footer={null}
      styles={{ body: { padding: 0 } }}
    >
      {/* Header */}
      <div className="px-6 pt-6 pb-3">
        <div className="flex items-center gap-2 mb-1">
          <svg className="w-5 h-5 text-blue-600" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 1l1.5 3.5L13 6l-3.5 1.5L8 11 6.5 7.5 3 6l3.5-1.5L8 1zm4 7l.75 1.75L14.5 10.5l-1.75.75L12 13l-.75-1.75L9.5 10.5l1.75-.75L12 8zM4 9l.75 1.75L6.5 11.5l-1.75.75L4 14l-.75-1.75L1.5 11.5l1.75-.75L4 9z" />
          </svg>
          <h3 className="text-lg font-semibold text-gray-900">Ask AI about Usage</h3>
        </div>
        <p className="text-sm text-gray-500">
          Ask questions about your spend, models, API keys, and usage trends
        </p>
      </div>

      <div className="border-t border-gray-100" />

      <div className="px-6 py-4 space-y-4">
        {/* Model selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Model
            <span className="text-red-500 ml-0.5">*</span>
          </label>
          <Select
            placeholder="Select a model"
            value={selectedModel}
            onChange={(value) => setSelectedModel(value)}
            loading={isLoadingModels}
            showSearch
            className="w-full"
            options={availableModels.map((m) => ({ label: m, value: m }))}
            filterOption={(input, option) =>
              (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
            }
          />
        </div>

        {/* Chat messages */}
        <div className="border border-gray-200 rounded-lg bg-gray-50 h-80 overflow-y-auto p-4 space-y-3">
          {messages.length === 0 && !streamingContent && (
            <div className="flex flex-col items-center justify-center h-full text-gray-400">
              <svg className="w-10 h-10 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
              </svg>
              <p className="text-sm font-medium">Ask a question about your usage</p>
              <p className="text-xs mt-1">e.g. &quot;Which model costs me the most?&quot;</p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "bg-blue-600 text-white"
                    : "bg-white border border-gray-200 text-gray-800"
                }`}
              >
                <div className="whitespace-pre-wrap break-words">{msg.content}</div>
              </div>
            </div>
          ))}

          {/* Streaming response */}
          {streamingContent && (
            <div className="flex justify-start">
              <div className="max-w-[85%] rounded-xl px-4 py-2.5 text-sm leading-relaxed bg-white border border-gray-200 text-gray-800">
                <div className="whitespace-pre-wrap break-words">{streamingContent}</div>
              </div>
            </div>
          )}

          {/* Loading indicator */}
          {isLoading && !streamingContent && (
            <div className="flex justify-start">
              <div className="rounded-xl px-4 py-2.5 bg-white border border-gray-200">
                <Spin size="small" />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input area */}
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
            onClick={handleSend}
            disabled={!inputText.trim() || !selectedModel || isLoading}
            loading={isLoading}
          >
            Send
          </Button>
        </div>

        {/* Footer actions */}
        <div className="flex justify-between items-center pt-1">
          <button
            onClick={handleClear}
            className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
            disabled={messages.length === 0}
          >
            Clear conversation
          </button>
          <span className="text-xs text-gray-400">
            Press Enter to send, Shift+Enter for new line
          </span>
        </div>
      </div>
    </Modal>
  );
};

export default UsageAIChatModal;
