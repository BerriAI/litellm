import React, { useEffect, useRef, useState } from "react";
import { Select, Input, Spin } from "antd";
import { Button } from "@tremor/react";
import { modelHubCall, usageAiChatStream } from "../../networking";

const { TextArea } = Input;

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface UsageAIChatPanelProps {
  open: boolean;
  onClose: () => void;
  accessToken: string | null;
}

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

    let accumulated = "";

    try {
      await usageAiChatStream(
        accessToken,
        updatedMessages.map((m) => ({ role: m.role, content: m.content })),
        selectedModel,
        (content: string) => {
          accumulated += content;
          setStreamingContent(accumulated);
        },
        () => {
          setMessages((prev) => [...prev, { role: "assistant", content: accumulated }]);
          setStreamingContent("");
        },
        (errorMsg: string) => {
          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: `Error: ${errorMsg}` },
          ]);
          setStreamingContent("");
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
          placeholder="Select a model"
          value={selectedModel}
          onChange={(value) => setSelectedModel(value)}
          loading={isLoadingModels}
          showSearch
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
        {messages.length === 0 && !streamingContent && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400">
            <svg className="w-8 h-8 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
              className={`max-w-[88%] rounded-xl px-3.5 py-2 text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-white border border-gray-200 text-gray-800"
              }`}
            >
              <div className="whitespace-pre-wrap break-words">{msg.content}</div>
            </div>
          </div>
        ))}

        {streamingContent && (
          <div className="flex justify-start">
            <div className="max-w-[88%] rounded-xl px-3.5 py-2 text-sm leading-relaxed bg-white border border-gray-200 text-gray-800">
              <div className="whitespace-pre-wrap break-words">{streamingContent}</div>
            </div>
          </div>
        )}

        {isLoading && !streamingContent && (
          <div className="flex justify-start">
            <div className="rounded-xl px-3.5 py-2 bg-white border border-gray-200">
              <Spin size="small" />
            </div>
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
            onClick={handleSend}
            disabled={!inputText.trim() || !selectedModel || isLoading}
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
