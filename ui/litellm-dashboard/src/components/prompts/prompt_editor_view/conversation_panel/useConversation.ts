import { useState, useRef, useEffect } from "react";
import NotificationsManager from "../../../molecules/notifications_manager";
import { TokenUsage } from "../../../playground/chat_ui/ResponseMetrics";
import { Message } from "./types";
import { convertToDotPrompt, extractVariables } from "../utils";
import { getProxyBaseUrl } from "../../../networking";

export const useConversation = (prompt: any, accessToken: string | null) => {
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [variablesFilled, setVariablesFilled] = useState(false);
  const [abortController, setAbortController] = useState<AbortController | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const extractedVariables = extractVariables(prompt);

  const allVariablesFilled = extractedVariables.every(
    (varName) => variables[varName] && variables[varName].trim() !== "",
  );

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "end",
        });
      }, 100);
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("Access token is required");
      return;
    }

    if (extractedVariables.length > 0 && !allVariablesFilled) {
      NotificationsManager.fromBackend("Please fill in all template variables");
      return;
    }

    if (!inputMessage.trim()) {
      return;
    }

    if (!variablesFilled && extractedVariables.length > 0) {
      setVariablesFilled(true);
    }

    const userMessage: Message = { role: "user", content: inputMessage };
    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");

    const controller = new AbortController();
    setAbortController(controller);
    setIsLoading(true);

    const startTime = Date.now();
    let timeToFirstToken: number | undefined;

    try {
      const dotpromptContent = convertToDotPrompt(prompt);
      const proxyBaseUrl = getProxyBaseUrl();

      const requestBody: any = {
        dotprompt_content: dotpromptContent,
      };

      if (messages.length === 0) {
        requestBody.prompt_variables = variables;
      } else {
        requestBody.conversation_history = [
          ...messages.map((msg) => ({
            role: msg.role,
            content: msg.content,
          })),
          {
            role: "user",
            content: inputMessage,
          },
        ];
      }

      const response = await fetch(`${proxyBaseUrl}/prompts/test`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP error! status: ${response.status}, ${errorText}`);
      }

      if (!response.body) {
        throw new Error("No response body");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let assistantMessage = "";
      let model: string | undefined;
      let usage: TokenUsage | undefined;
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data === "[DONE]") {
              continue;
            }

            try {
              const parsed = JSON.parse(data);

              if (!model && parsed.model) {
                model = parsed.model;
              }

              if (parsed.usage) {
                usage = parsed.usage;
              }

              const content = parsed.choices?.[0]?.delta?.content;
              if (content) {
                if (!timeToFirstToken) {
                  timeToFirstToken = Date.now() - startTime;
                }
                assistantMessage += content;
                setMessages((prev) => {
                  const newMessages = [...prev];
                  newMessages[newMessages.length - 1] = {
                    role: "assistant",
                    content: assistantMessage,
                    model,
                    timeToFirstToken,
                  };
                  return newMessages;
                });
              }
            } catch (e) {
              console.error("Error parsing chunk:", e);
            }
          }
        }
      }

      const totalLatency = Date.now() - startTime;
      setMessages((prev) => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1] = {
          ...newMessages[newMessages.length - 1],
          totalLatency,
          usage,
        };
        return newMessages;
      });
    } catch (error: any) {
      if (error.name === "AbortError") {
        console.log("Request was cancelled");
      } else {
        console.error("Error testing prompt:", error);
        setMessages((prev) => {
          const lastMsg = prev[prev.length - 1];
          if (lastMsg && lastMsg.role === "assistant" && lastMsg.content === "") {
            return [...prev.slice(0, -1), { role: "assistant", content: `Error: ${error.message}` }];
          }
          return [...prev, { role: "assistant", content: `Error: ${error.message}` }];
        });
      }
    } finally {
      setIsLoading(false);
      setAbortController(null);
    }
  };

  const handleCancelRequest = () => {
    if (abortController) {
      abortController.abort();
      setAbortController(null);
      setIsLoading(false);
      NotificationsManager.info("Request cancelled");
    }
  };

  const handleClearConversation = () => {
    setMessages([]);
    setVariablesFilled(false);
    NotificationsManager.success("Chat history cleared.");
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSendMessage();
    }
  };

  const handleVariableChange = (varName: string, value: string) => {
    setVariables({ ...variables, [varName]: value });
  };

  return {
    // State
    isLoading,
    messages,
    inputMessage,
    variables,
    variablesFilled,
    extractedVariables,
    allVariablesFilled,
    messagesEndRef,

    // Actions
    setInputMessage,
    handleSendMessage,
    handleCancelRequest,
    handleClearConversation,
    handleKeyDown,
    handleVariableChange,
  };
};
