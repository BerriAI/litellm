"use client";

import NotificationsManager from "@/components/molecules/notifications_manager";
import { ClearOutlined, DeleteOutlined, FilePdfOutlined, PlusOutlined } from "@ant-design/icons";
import { Button, Input, Select, Tooltip } from "antd";
import { useEffect, useMemo, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import ChatImageUpload from "../chat_ui/ChatImageUpload";
import { createChatDisplayMessage, createChatMultimodalMessage } from "../chat_ui/ChatImageUtils";
import type { TokenUsage } from "../chat_ui/ResponseMetrics";
import type { MessageType, VectorStoreSearchResponse } from "../chat_ui/types";
import { makeOpenAIChatCompletionRequest } from "../llm_calls/chat_completion";
import { fetchAvailableModels } from "../llm_calls/fetch_models";
import { Agent, fetchAvailableAgents } from "../llm_calls/fetch_agents";
import { makeA2AStreamMessageRequest } from "../llm_calls/a2a_send_message";
import { ComparisonPanel } from "./components/ComparisonPanel";
import { MessageInput } from "./components/MessageInput";
import {
  EndpointId,
  EndpointIdType,
  getAvailableEndpoints,
  getEndpointConfig,
  isAgentEndpoint,
  hasValidSelection,
  modelOptionsToSelectorOptions,
  agentOptionsToSelectorOptions,
} from "./endpoint_config";
export interface ComparisonInstance {
  id: string;
  model: string;
  agent: string;
  messages: MessageType[];
  isLoading: boolean;
  tags: string[];
  mcpTools: string[];
  vectorStores: string[];
  guardrails: string[];
  temperature: number;
  maxTokens: number;
  applyAcrossModels: boolean;
  useAdvancedParams: boolean;
  traceId?: string;
}
interface CompareUIProps {
  accessToken: string | null;
  disabledPersonalKeyCreation: boolean;
}
const GENERIC_FOLLOW_UPS = [
  "Can you summarize the key points?",
  "What assumptions did you make?",
  "What are the next steps?",
];
const SUGGESTED_PROMPTS = ["Write me a poem", "Explain quantum computing", "Draft a polite email requesting a meeting"];
const DEFAULT_ENDPOINT = EndpointId.CHAT_COMPLETIONS;
export default function CompareUI({ accessToken, disabledPersonalKeyCreation }: CompareUIProps) {
  const [comparisons, setComparisons] = useState<ComparisonInstance[]>([
    {
      id: "1",
      model: "",
      agent: "",
      messages: [],
      isLoading: false,
      tags: [],
      mcpTools: [],
      vectorStores: [],
      guardrails: [],
      temperature: 1,
      maxTokens: 2048,
      applyAcrossModels: false,
      useAdvancedParams: false,
    },
    {
      id: "2",
      model: "",
      agent: "",
      messages: [],
      isLoading: false,
      tags: [],
      mcpTools: [],
      vectorStores: [],
      guardrails: [],
      temperature: 1,
      maxTokens: 2048,
      applyAcrossModels: false,
      useAdvancedParams: false,
    },
  ]);
  const [modelOptions, setModelOptions] = useState<string[]>([]);
  const [agentOptions, setAgentOptions] = useState<Agent[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [isLoadingAgents, setIsLoadingAgents] = useState(false);
  const [selectedEndpoint, setSelectedEndpoint] = useState<EndpointIdType>(DEFAULT_ENDPOINT);
  
  // Derived state from endpoint config
  const endpointConfig = getEndpointConfig(selectedEndpoint);
  const isA2AMode = isAgentEndpoint(selectedEndpoint);
  const selectorOptions = isA2AMode
    ? agentOptionsToSelectorOptions(agentOptions)
    : modelOptionsToSelectorOptions(modelOptions);
  const isLoadingOptions = isA2AMode ? isLoadingAgents : isLoadingModels;
  const [inputValue, setInputValue] = useState("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedFilePreviewUrl, setUploadedFilePreviewUrl] = useState<string | null>(null);
  const [apiKeySource, setApiKeySource] = useState<"session" | "custom">(
    disabledPersonalKeyCreation ? "custom" : "session",
  );
  const [customApiKey, setCustomApiKey] = useState("");
  const [debouncedCustomApiKey, setDebouncedCustomApiKey] = useState("");
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedCustomApiKey(customApiKey);
    }, 300);
    return () => clearTimeout(timer);
  }, [customApiKey]);
  useEffect(() => {
    return () => {
      if (uploadedFilePreviewUrl) {
        URL.revokeObjectURL(uploadedFilePreviewUrl);
      }
    };
  }, [uploadedFilePreviewUrl]);
  const effectiveApiKey = useMemo(
    () => (apiKeySource === "session" ? accessToken || "" : debouncedCustomApiKey.trim()),
    [apiKeySource, accessToken, debouncedCustomApiKey],
  );
  const haveAllResponses = useMemo(
    () =>
      comparisons.length > 0 &&
      comparisons.every(
        (comparison) => !comparison.isLoading && comparison.messages.some((message) => message.role === "assistant"),
      ),
    [comparisons],
  );
  useEffect(() => {
    let active = true;
    const loadModels = async () => {
      if (!effectiveApiKey) {
        setModelOptions([]);
        return;
      }
      setIsLoadingModels(true);
      try {
        const uniqueModels = await fetchAvailableModels(effectiveApiKey);
        if (!active) return;
        const nextOptions = Array.from(new Set(uniqueModels.map((model) => model.model_group)));
        setModelOptions(nextOptions);
      } catch (error) {
        console.error("CompareUI: failed to fetch models", error);
        if (active) {
          setModelOptions([]);
        }
      } finally {
        if (active) {
          setIsLoadingModels(false);
        }
      }
    };
    loadModels();
    return () => {
      active = false;
    };
  }, [effectiveApiKey]);

  // Fetch agents when A2A mode is selected
  useEffect(() => {
    let active = true;
    const loadAgents = async () => {
      if (!effectiveApiKey || !isA2AMode) {
        setAgentOptions([]);
        return;
      }
      setIsLoadingAgents(true);
      try {
        const agents = await fetchAvailableAgents(effectiveApiKey);
        if (!active) return;
        setAgentOptions(agents);
      } catch (error) {
        console.error("CompareUI: failed to fetch agents", error);
        if (active) {
          setAgentOptions([]);
        }
      } finally {
        if (active) {
          setIsLoadingAgents(false);
        }
      }
    };
    loadAgents();
    return () => {
      active = false;
    };
  }, [effectiveApiKey, isA2AMode]);

  useEffect(() => {
    if (modelOptions.length === 0) {
      return;
    }
    setComparisons((prev) =>
      prev.map((comparison, index) => {
        return {
          ...comparison,
          temperature: comparison.temperature ?? 1,
          maxTokens: comparison.maxTokens ?? 2048,
          applyAcrossModels: comparison.applyAcrossModels ?? false,
          useAdvancedParams: comparison.useAdvancedParams ?? false,
          ...(comparison.model
            ? {}
            : {
                model: modelOptions[index % modelOptions.length] ?? "",
              }),
        };
      }),
    );
  }, [modelOptions]);
  const maxComparisons = 3;
  const addComparison = () => {
    if (comparisons.length >= maxComparisons) {
      return;
    }
    const fallbackModel = modelOptions[comparisons.length % (modelOptions.length || 1)] ?? "";
    const fallbackAgent = agentOptions[comparisons.length % (agentOptions.length || 1)]?.agent_name ?? "";
    const newComparison: ComparisonInstance = {
      id: Date.now().toString(),
      model: fallbackModel,
      agent: fallbackAgent,
      messages: [],
      isLoading: false,
      tags: [],
      mcpTools: [],
      vectorStores: [],
      guardrails: [],
      temperature: 1,
      maxTokens: 2048,
      applyAcrossModels: false,
      useAdvancedParams: false,
    };
    setComparisons((prev) => [...prev, newComparison]);
  };
  const removeComparison = (id: string) => {
    if (comparisons.length > 1) {
      setComparisons((prev) => {
        const next = prev.filter((c) => c.id !== id);
        return next;
      });
    }
  };
  type UpdateOptions = {
    applyToAll?: boolean;
    keysToApply?: (keyof ComparisonInstance)[];
  };
  const updateComparison = (id: string, updates: Partial<ComparisonInstance>, options?: UpdateOptions) => {
    setComparisons((prev) => {
      if (options?.applyToAll && options.keysToApply?.length) {
        const sharedUpdates: Partial<ComparisonInstance> = {};
        options.keysToApply.forEach((key) => {
          const value = updates[key];
          if (value !== undefined) {
            sharedUpdates[key] = Array.isArray(value) ? ([...value] as any) : (value as any);
          }
        });
        const hasSharedUpdates = Object.keys(sharedUpdates).length > 0;
        return prev.map((comparison) => {
          if (comparison.id === id) {
            return {
              ...comparison,
              ...updates,
            };
          }
          if (!hasSharedUpdates) {
            return comparison;
          }
          return {
            ...comparison,
            ...sharedUpdates,
          };
        });
      }
      return prev.map((comparison) =>
        comparison.id === id
          ? {
              ...comparison,
              ...updates,
            }
          : comparison,
      );
    });
  };
  const handleFileUpload = (file: File): false => {
    if (uploadedFilePreviewUrl) {
      URL.revokeObjectURL(uploadedFilePreviewUrl);
    }
    setUploadedFile(file);
    setUploadedFilePreviewUrl(URL.createObjectURL(file));
    return false;
  };
  const handleRemoveFile = () => {
    if (uploadedFilePreviewUrl) {
      URL.revokeObjectURL(uploadedFilePreviewUrl);
    }
    setUploadedFile(null);
    setUploadedFilePreviewUrl(null);
  };
  const clearAllChats = () => {
    setComparisons((prev) =>
      prev.map((comparison) => ({
        ...comparison,
        messages: [],
        traceId: undefined,
        isLoading: false,
      })),
    );
    setInputValue("");
    handleRemoveFile();
  };
  const appendAssistantChunk = (comparisonId: string, chunk: string, model?: string) => {
    if (!chunk) {
      return;
    }
    setComparisons((prev) =>
      prev.map((comparison) => {
        if (comparison.id !== comparisonId) {
          return comparison;
        }
        const messages = [...comparison.messages];
        const last = messages[messages.length - 1];
        if (last && last.role === "assistant") {
          const existingContent = typeof last.content === "string" ? last.content : "";
          messages[messages.length - 1] = {
            ...last,
            content: existingContent + chunk,
            model: last.model ?? model,
          };
        } else {
          messages.push({
            role: "assistant",
            content: chunk,
            model,
          });
        }
        return {
          ...comparison,
          messages,
        };
      }),
    );
  };
  const appendReasoningContent = (comparisonId: string, chunk: string) => {
    if (!chunk) {
      return;
    }
    setComparisons((prev) =>
      prev.map((comparison) => {
        if (comparison.id !== comparisonId) {
          return comparison;
        }
        const messages = [...comparison.messages];
        const last = messages[messages.length - 1];
        if (last && last.role === "assistant") {
          messages[messages.length - 1] = {
            ...last,
            reasoningContent: (last.reasoningContent || "") + chunk,
          };
        } else if (last && last.role === "user") {
          messages.push({
            role: "assistant",
            content: "",
            reasoningContent: chunk,
          });
        }
        return {
          ...comparison,
          messages,
        };
      }),
    );
  };
  const updateTimingDataForComparison = (comparisonId: string, timeToFirstToken: number) => {
    setComparisons((prev) =>
      prev.map((comparison) => {
        if (comparison.id !== comparisonId) {
          return comparison;
        }
        const messages = [...comparison.messages];
        const last = messages[messages.length - 1];
        if (last && last.role === "assistant") {
          messages[messages.length - 1] = {
            ...last,
            timeToFirstToken,
          };
        } else if (last && last.role === "user") {
          messages.push({
            role: "assistant",
            content: "",
            timeToFirstToken,
          });
        }
        return {
          ...comparison,
          messages,
        };
      }),
    );
  };
  const updateTotalLatencyForComparison = (comparisonId: string, totalLatency: number) => {
    setComparisons((prev) =>
      prev.map((comparison) => {
        if (comparison.id !== comparisonId) {
          return comparison;
        }
        const messages = [...comparison.messages];
        const last = messages[messages.length - 1];
        if (last && last.role === "assistant") {
          messages[messages.length - 1] = {
            ...last,
            totalLatency,
          };
        } else if (last && last.role === "user") {
          messages.push({
            role: "assistant",
            content: "",
            totalLatency,
          });
        }
        return {
          ...comparison,
          messages,
        };
      }),
    );
  };
  const updateUsageDataForComparison = (comparisonId: string, usage: TokenUsage, toolName?: string) => {
    setComparisons((prev) =>
      prev.map((comparison) => {
        if (comparison.id !== comparisonId) {
          return comparison;
        }
        const messages = [...comparison.messages];
        const last = messages[messages.length - 1];
        if (last && last.role === "assistant") {
          messages[messages.length - 1] = {
            ...last,
            usage,
            toolName,
          };
        }
        return {
          ...comparison,
          messages,
        };
      }),
    );
  };
  const updateSearchResultsForComparison = (comparisonId: string, searchResults: VectorStoreSearchResponse[]) => {
    if (!searchResults) {
      return;
    }
    setComparisons((prev) =>
      prev.map((comparison) => {
        if (comparison.id !== comparisonId) {
          return comparison;
        }
        const messages = [...comparison.messages];
        const last = messages[messages.length - 1];
        if (last && last.role === "assistant") {
          messages[messages.length - 1] = {
            ...last,
            searchResults,
          };
        }
        return {
          ...comparison,
          messages,
        };
      }),
    );
  };
  const canUseSessionKey = Boolean(accessToken);
  const handleSendMessage = async (input: string) => {
    const trimmed = input.trim();
    const hasAttachment = Boolean(uploadedFile);
    if (!trimmed && !hasAttachment) {
      return;
    }
    if (!effectiveApiKey) {
      NotificationsManager.fromBackend("Please provide a Virtual Key or select Current UI Session");
      return;
    }
    const targetComparisons = comparisons;
    if (targetComparisons.length === 0) {
      return;
    }
    // Validate selection based on endpoint type
    if (targetComparisons.some((comparison) => !hasValidSelection(comparison, selectedEndpoint))) {
      NotificationsManager.fromBackend(endpointConfig.validationMessage);
      return;
    }

    const apiUserMessage = hasAttachment
      ? await createChatMultimodalMessage(trimmed, uploadedFile as File)
      : { role: "user", content: trimmed };
    const displayUserMessage = createChatDisplayMessage(
      trimmed,
      hasAttachment,
      uploadedFilePreviewUrl || undefined,
      uploadedFile?.name,
    );

    const preparedTargets = new Map<
      string,
      {
        id: string;
        model: string;
        agent: string;
        inputMessage: string;
        traceId: string;
        tags: string[];
        vectorStores: string[];
        guardrails: string[];
        temperature: number;
        maxTokens: number;
        displayMessages: MessageType[];
        apiChatHistory: Array<{ role: string; content: string | any[] }>;
      }
    >();
    targetComparisons.forEach((comparison) => {
      const traceId = comparison.traceId ?? uuidv4();
      const apiChatHistory = [
        ...comparison.messages.map(({ role, content }) => ({
          role,
          content: Array.isArray(content) ? content : typeof content === "string" ? content : "",
        })),
        apiUserMessage,
      ];
      preparedTargets.set(comparison.id, {
        id: comparison.id,
        model: comparison.model,
        agent: comparison.agent,
        inputMessage: trimmed,
        traceId,
        tags: comparison.tags,
        vectorStores: comparison.vectorStores,
        guardrails: comparison.guardrails,
        temperature: comparison.temperature,
        maxTokens: comparison.maxTokens,
        displayMessages: [...comparison.messages, displayUserMessage],
        apiChatHistory,
      });
    });
    if (preparedTargets.size === 0) {
      return;
    }
    setComparisons((prev) =>
      prev.map((comparison) => {
        const prepared = preparedTargets.get(comparison.id);
        if (!prepared) {
          return comparison;
        }
        return {
          ...comparison,
          traceId: prepared.traceId,
          messages: prepared.displayMessages,
          isLoading: true,
        };
      }),
    );
    setInputValue("");
    handleRemoveFile();

    preparedTargets.forEach((prepared) => {
      const tags = prepared.tags.length > 0 ? prepared.tags : undefined;
      const vectorStoreIds = prepared.vectorStores.length > 0 ? prepared.vectorStores : undefined;
      const guardrails = prepared.guardrails.length > 0 ? prepared.guardrails : undefined;
      const comparison = comparisons.find((c) => c.id === prepared.id);
      const useAdvancedParams = comparison?.useAdvancedParams ?? false;

      // Use A2A or chat completion based on endpoint
      const requestPromise = isA2AMode
        ? makeA2AStreamMessageRequest(
            prepared.agent,
            prepared.inputMessage,
            (text, model) => {
              // A2A sends full accumulated text, so replace instead of append
              setComparisons((prev) =>
                prev.map((c) => {
                  if (c.id !== prepared.id) return c;
                  const messages = [...c.messages];
                  const last = messages[messages.length - 1];
                  if (last && last.role === "assistant") {
                    messages[messages.length - 1] = { ...last, content: text, model: last.model ?? model };
                  } else {
                    messages.push({ role: "assistant", content: text, model });
                  }
                  return { ...c, messages };
                }),
              );
            },
            effectiveApiKey,
            undefined,
            (time) => updateTimingDataForComparison(prepared.id, time),
            (latency) => updateTotalLatencyForComparison(prepared.id, latency),
          )
        : makeOpenAIChatCompletionRequest(
            prepared.apiChatHistory,
            (chunk, model) => appendAssistantChunk(prepared.id, chunk, model),
            prepared.model,
            effectiveApiKey,
            tags,
            undefined,
            (content) => appendReasoningContent(prepared.id, content),
            (time) => updateTimingDataForComparison(prepared.id, time),
            (usage) => updateUsageDataForComparison(prepared.id, usage),
            prepared.traceId,
            vectorStoreIds,
            guardrails,
            undefined,
            undefined,
            (searchResults) => updateSearchResultsForComparison(prepared.id, searchResults),
            useAdvancedParams ? prepared.temperature : undefined,
            useAdvancedParams ? prepared.maxTokens : undefined,
            (latency) => updateTotalLatencyForComparison(prepared.id, latency),
          );

      requestPromise
        .catch((error) => {
          const errorMessage = error instanceof Error ? error.message : String(error);
          console.error("CompareUI: failed to fetch response", error);
          NotificationsManager.fromBackend(errorMessage);
          setComparisons((prev) =>
            prev.map((comparison) => {
              if (comparison.id !== prepared.id) {
                return comparison;
              }
              const messages = [...comparison.messages];
              const last = messages[messages.length - 1];
              const assistantContent =
                last && last.role === "assistant" && typeof last.content === "string" ? last.content : "";
              if (last && last.role === "assistant") {
                messages[messages.length - 1] = {
                  ...last,
                  content: assistantContent
                    ? `${assistantContent}\nError fetching response: ${errorMessage}`
                    : `Error fetching response: ${errorMessage}`,
                };
              } else {
                messages.push({
                  role: "assistant",
                  content: `Error fetching response: ${errorMessage}`,
                });
              }
              return {
                ...comparison,
                messages,
              };
            }),
          );
        })
        .finally(() => {
          setComparisons((prev) =>
            prev.map((comparison) =>
              comparison.id === prepared.id
                ? {
                    ...comparison,
                    isLoading: false,
                  }
                : comparison,
            ),
          );
        });
    });
  };
  const handleInputChange = (value: string) => {
    setInputValue(value);
  };
  const handleSubmit = () => {
    void handleSendMessage(inputValue);
  };
  const handleFollowUpSelect = (question: string) => {
    setInputValue(question);
  };
  const hasMessages = comparisons.some((comparison) => comparison.messages.length > 0);
  const isAnyComparisonLoading = comparisons.some((comparison) => comparison.isLoading);
  const hasAttachment = Boolean(uploadedFile);
  const isUploadedFilePdf = Boolean(uploadedFile?.name.toLowerCase().endsWith(".pdf"));
  const showSuggestedPrompts = !hasMessages && !isAnyComparisonLoading && !hasAttachment;
  return (
    <div className="w-full h-full p-4 bg-white">
      <div className="rounded-2xl border border-gray-200 bg-white shadow-sm min-h-[calc(100vh-160px)] flex flex-col">
        <div className="border-b px-4 py-2">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-600">Virtual Key Source</span>
              <Select
                value={apiKeySource}
                onChange={(value) => setApiKeySource(value as "session" | "custom")}
                disabled={disabledPersonalKeyCreation}
                className="w-48"
              >
                <Select.Option value="session" disabled={!canUseSessionKey}>
                  Current UI Session
                </Select.Option>
                <Select.Option value="custom">Virtual Key</Select.Option>
              </Select>
              {apiKeySource === "custom" && (
                <Input.Password
                  value={customApiKey}
                  onChange={(event) => setCustomApiKey(event.target.value)}
                  placeholder="Enter Virtual Key"
                  className="w-56"
                />
              )}
            </div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-600">Endpoint</span>
              <Select 
                value={selectedEndpoint} 
                onChange={(value) => setSelectedEndpoint(value as EndpointIdType)}
                className="w-56"
              >
                {getAvailableEndpoints().map((endpoint) => (
                  <Select.Option 
                    key={endpoint.value} 
                    value={endpoint.value}
                  >
                    {endpoint.label}
                  </Select.Option>
                ))}
              </Select>
            </div>
            <div className="flex items-center gap-3">
              <Button onClick={clearAllChats} disabled={!hasMessages} icon={<ClearOutlined />}>
                Clear All Chats
              </Button>
              <Tooltip
                title={
                  comparisons.length >= maxComparisons ? "Compare up to 3 models at a time" : "Add another comparison"
                }
              >
                <Button onClick={addComparison} disabled={comparisons.length >= maxComparisons} icon={<PlusOutlined />}>
                  Add Comparison
                </Button>
              </Tooltip>
            </div>
          </div>
        </div>

        <div
          className="grid flex-1 min-h-0 auto-rows-[minmax(0,1fr)]"
          style={{
            gridTemplateColumns: `repeat(${comparisons.length}, minmax(0, 1fr))`,
          }}
        >
          {comparisons.map((comparison) => (
            <ComparisonPanel
              key={comparison.id}
              comparison={comparison}
              onUpdate={(updates, options) => updateComparison(comparison.id, updates, options)}
              onRemove={() => removeComparison(comparison.id)}
              canRemove={comparisons.length > 1}
              selectorOptions={selectorOptions}
              isLoadingOptions={isLoadingOptions}
              endpointConfig={endpointConfig}
              apiKey={effectiveApiKey}
            />
          ))}
        </div>
        <div className="flex justify-center pb-4">
          <div className="w-full max-w-3xl px-4">
            <div className="border border-gray-200 shadow-lg rounded-xl bg-white p-4">
              <div className="flex items-center justify-between gap-4 mb-3 min-h-8">
                {hasAttachment ? (
                  <span className="text-sm text-gray-500">Attachment ready to send</span>
                ) : showSuggestedPrompts ? (
                  <div className="flex items-center gap-2 overflow-x-auto">
                    {SUGGESTED_PROMPTS.map((prompt) => (
                      <button
                        key={prompt}
                        type="button"
                        onClick={() => handleFollowUpSelect(prompt)}
                        className="shrink-0 rounded-full border border-gray-200 px-3 py-1 text-xs font-medium text-gray-600 transition-colors hover:bg-gray-100 cursor-pointer"
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                ) : haveAllResponses && !hasAttachment ? (
                  <div className="flex items-center gap-2 overflow-x-auto">
                    {GENERIC_FOLLOW_UPS.map((question) => (
                      <button
                        key={question}
                        type="button"
                        onClick={() => handleFollowUpSelect(question)}
                        className="shrink-0 rounded-full border border-gray-200 px-3 py-1 text-xs font-medium text-gray-600 transition-colors hover:bg-gray-100 cursor-pointer"
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                ) : isAnyComparisonLoading ? (
                  <span className="flex items-center gap-2 text-sm text-gray-500">
                    <span className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" aria-hidden />
                    {endpointConfig.loadingMessage}
                  </span>
                ) : (
                  <span className="text-sm text-gray-500">{endpointConfig.inputPlaceholder}</span>
                )}
              </div>
              {uploadedFile && (
                <div className="mb-3">
                  <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                    <div className="relative inline-block">
                      {isUploadedFilePdf ? (
                        <div className="w-10 h-10 rounded-md bg-red-500 flex items-center justify-center">
                          <FilePdfOutlined style={{ fontSize: "16px", color: "white" }} />
                        </div>
                      ) : (
                        <img
                          src={uploadedFilePreviewUrl || ""}
                          alt="Upload preview"
                          className="w-10 h-10 rounded-md border border-gray-200 object-cover"
                        />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-gray-900 truncate">{uploadedFile.name}</div>
                      <div className="text-xs text-gray-500">{isUploadedFilePdf ? "PDF" : "Image"}</div>
                    </div>
                    <button
                      className="flex items-center justify-center w-6 h-6 text-gray-400 hover:text-gray-600 hover:bg-gray-200 rounded-full transition-colors"
                      onClick={handleRemoveFile}
                    >
                      <DeleteOutlined style={{ fontSize: "12px" }} />
                    </button>
                  </div>
                </div>
              )}
              <MessageInput
                value={inputValue}
                onChange={handleInputChange}
                onSend={handleSubmit}
                disabled={comparisons.length === 0 || comparisons.every((comparison) => comparison.isLoading)}
                hasAttachment={hasAttachment}
                uploadComponent={
                  <ChatImageUpload
                    chatUploadedImage={uploadedFile}
                    chatImagePreviewUrl={uploadedFilePreviewUrl}
                    onImageUpload={handleFileUpload}
                    onRemoveImage={handleRemoveFile}
                  />
                }
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
