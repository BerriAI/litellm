import React, { useState, useEffect, useRef } from "react";
import { Button as TremorButton, Card, Text, Title } from "@tremor/react";
import { Input, Select, Spin, Modal } from "antd";
import {
  PlusIcon,
  MoreHorizontalIcon,
  FlaskConicalIcon,
  SaveIcon,
  MessageSquareIcon,
  SettingsIcon,
  TrashIcon,
  ArrowLeftIcon,
  SendIcon,
} from "lucide-react";
import { RobotOutlined, UserOutlined, LoadingOutlined, ClearOutlined } from "@ant-design/icons";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import ToolModal from "./tool_modal";
import VariableTextArea from "./variable_textarea";
import ModelSelector from "../common_components/ModelSelector";
import AdditionalModelSettings from "../chat_ui/AdditionalModelSettings";
import NotificationsManager from "../molecules/notifications_manager";
import { createPromptCall } from "../networking";
import ResponseMetrics from "../chat_ui/ResponseMetrics";
import ReasoningContent from "../chat_ui/ReasoningContent";
import { MessageType } from "../chat_ui/types";
import { makeOpenAIChatCompletionRequest } from "../chat_ui/llm_calls/chat_completion";

const { TextArea } = Input;
const { Option } = Select;

interface Message {
  role: string;
  content: string;
}

interface Tool {
  name: string;
  description: string;
  json: string;
}

interface PromptType {
  name: string;
  model: string;
  config: {
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
  };
  tools: Tool[];
  developerMessage: string;
  messages: Message[];
}

interface PromptEditorViewProps {
  onClose: () => void;
  onSuccess: () => void;
  accessToken: string | null;
}

const PromptEditorView: React.FC<PromptEditorViewProps> = ({ onClose, onSuccess, accessToken }) => {
  const [prompt, setPrompt] = useState<PromptType>({
    name: "New prompt",
    model: "gpt-4o",
    config: {
      temperature: 1,
      max_tokens: 1000,
    },
    tools: [],
    developerMessage: "",
    messages: [
      {
        role: "user",
        content: "Enter task specifics. Use {{template_variables}} for dynamic inputs",
      },
    ],
  });

  // Extract variables from all messages
  const extractVariables = (): string[] => {
    const variableSet = new Set<string>();
    const variableRegex = /\{\{(\w+)\}\}/g;

    prompt.messages.forEach((message) => {
      let match;
      while ((match = variableRegex.exec(message.content)) !== null) {
        variableSet.add(match[1]);
      }
    });

    if (prompt.developerMessage) {
      let match;
      while ((match = variableRegex.exec(prompt.developerMessage)) !== null) {
        variableSet.add(match[1]);
      }
    }

    return Array.from(variableSet);
  };

  const [showConfig, setShowConfig] = useState(false);
  const [showToolModal, setShowToolModal] = useState(false);
  const [showNameModal, setShowNameModal] = useState(false);
  const [editingToolIndex, setEditingToolIndex] = useState<number | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  const addMessage = () => {
    setPrompt({
      ...prompt,
      messages: [
        ...prompt.messages,
        {
          role: "user",
          content: "",
        },
      ],
    });
  };

  const updateMessage = (index: number, field: "role" | "content", value: string) => {
    const newMessages = [...prompt.messages];
    newMessages[index][field] = value;
    setPrompt({
      ...prompt,
      messages: newMessages,
    });
  };

  const removeMessage = (index: number) => {
    if (prompt.messages.length > 1) {
      setPrompt({
        ...prompt,
        messages: prompt.messages.filter((_, i) => i !== index),
      });
    }
  };


  const addTool = (json: string) => {
    try {
      const parsed = JSON.parse(json);
      const tool = {
        name: parsed.function?.name || "Unnamed Tool",
        description: parsed.function?.description || "",
        json: json,
      };

      if (editingToolIndex !== null) {
        const newTools = [...prompt.tools];
        newTools[editingToolIndex] = tool;
        setPrompt({
          ...prompt,
          tools: newTools,
        });
      } else {
        setPrompt({
          ...prompt,
          tools: [...prompt.tools, tool],
        });
      }

      setShowToolModal(false);
      setEditingToolIndex(null);
    } catch (error) {
      NotificationsManager.fromBackend("Invalid JSON format");
    }
  };

  const removeTool = (index: number) => {
    setPrompt({
      ...prompt,
      tools: prompt.tools.filter((_, i) => i !== index),
    });
  };

  const openToolModal = (index?: number) => {
    if (index !== undefined) {
      setEditingToolIndex(index);
    } else {
      setEditingToolIndex(null);
    }
    setShowToolModal(true);
  };

  const handleSaveClick = () => {
    // If name is default or empty, show the name modal
    if (!prompt.name || prompt.name.trim() === "" || prompt.name === "New prompt") {
      setShowNameModal(true);
    } else {
      // Otherwise proceed to save directly
      handleSave();
    }
  };

  const handleSave = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("Access token is required");
      return;
    }

    if (!prompt.name || prompt.name.trim() === "") {
      NotificationsManager.fromBackend("Please enter a valid prompt name");
      return;
    }

    setIsSaving(true);
    try {
      // Extract variables from messages
      const variables = extractVariables();

      // Convert the prompt to dotprompt format
      const dotPromptContent = convertToDotPrompt(prompt);

      // Create the prompt
      const promptData = {
        prompt_id: prompt.name.replace(/[^a-zA-Z0-9_-]/g, "_").toLowerCase(),
        litellm_params: {
          prompt_integration: "dotprompt",
          prompt_id: prompt.name,
          prompt_data: {
            model: prompt.model,
            input: {
              schema: variables.reduce((acc, v) => {
                acc[v] = "string";
                return acc;
              }, {} as Record<string, string>),
            },
            output: {
              format: "text",
            },
            messages: prompt.messages,
            tools: prompt.tools.map((t) => JSON.parse(t.json)),
            config: prompt.config,
          },
        },
        prompt_info: {
          prompt_type: "db",
        },
      };

      await createPromptCall(accessToken, promptData);
      NotificationsManager.success("Prompt created successfully!");
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Error saving prompt:", error);
      NotificationsManager.fromBackend("Failed to save prompt");
    } finally {
      setIsSaving(false);
      setShowNameModal(false);
    }
  };

  const convertToDotPrompt = (prompt: PromptType): string => {
    const variables = extractVariables();
    let result = `---\nmodel: ${prompt.model}\n\ninput:\n  schema:\n`;

    variables.forEach((variable) => {
      result += `    ${variable}: string\n`;
    });

    result += `\noutput:\n  format: text\n---\n\n`;

    prompt.messages.forEach((message) => {
      result += `${message.role}: ${message.content}\n\n`;
    });

    return result;
  };

  return (
    <div className="flex h-full bg-white">
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <TremorButton icon={ArrowLeftIcon} variant="light" onClick={onClose} size="xs">
              Back
            </TremorButton>
            <Input
              value={prompt.name}
              onChange={(e) =>
                setPrompt({
                  ...prompt,
                  name: e.target.value,
                })
              }
              className="text-base font-medium border-none shadow-none"
              style={{ width: "200px" }}
            />
            <span className="px-2 py-0.5 text-xs bg-gray-100 text-gray-600 rounded">Draft</span>
            <span className="text-xs text-gray-400">Unsaved changes</span>
          </div>
          <div className="flex items-center space-x-2">
            <TremorButton
              icon={SaveIcon}
              onClick={handleSaveClick}
              loading={isSaving}
              disabled={isSaving}
            >
              Save
            </TremorButton>
          </div>
        </div>

        <div className="flex-1 flex overflow-hidden">
          {/* Left Panel - Editor */}
          <div className="w-1/2 overflow-y-auto bg-white border-r border-gray-200">
            <div className="p-6 space-y-4 pb-20">
              {/* Model Card */}
              <Card className="p-3">
                <div className="mb-3">
                  <Text className="block mb-2 text-sm font-medium">Model</Text>
                  <ModelSelector
                    accessToken={accessToken || ""}
                    value={prompt.model}
                    onChange={(model) =>
                      setPrompt({
                        ...prompt,
                        model,
                      })
                    }
                    showLabel={false}
                  />
                </div>

                <button
                  onClick={() => setShowConfig(!showConfig)}
                  className="flex items-center text-xs font-medium text-gray-600 hover:text-gray-900"
                >
                  <SettingsIcon size={14} className="mr-1" />
                  <span>Configuration</span>
                </button>

                {showConfig && (
                  <div className="mt-3 pt-3 border-t border-gray-200 -mx-3 px-0">
                    <div className="space-y-3 px-3">
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <Text className="text-xs text-gray-700">Temperature</Text>
                          <Input
                            type="number"
                            size="small"
                            min={0}
                            max={2}
                            step={0.1}
                            value={prompt.config.temperature}
                            onChange={(e) =>
                              setPrompt({
                                ...prompt,
                                config: {
                                  ...prompt.config,
                                  temperature: parseFloat(e.target.value) || 0,
                                },
                              })
                            }
                            className="w-16"
                          />
                        </div>
                      </div>
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <Text className="text-xs text-gray-700">Max Tokens</Text>
                          <Input
                            type="number"
                            size="small"
                            min={1}
                            max={32768}
                            value={prompt.config.max_tokens}
                            onChange={(e) =>
                              setPrompt({
                                ...prompt,
                                config: {
                                  ...prompt.config,
                                  max_tokens: parseInt(e.target.value) || 1000,
                                },
                              })
                            }
                            className="w-20"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </Card>

              {/* Tools Card */}
              <Card className="p-3">
                <div className="flex items-center justify-between mb-2">
                  <Text className="text-sm font-medium">Tools</Text>
                  <button
                    onClick={() => openToolModal()}
                    className="text-xs text-blue-600 hover:text-blue-700 flex items-center"
                  >
                    <PlusIcon size={14} className="mr-1" />
                    Add
                  </button>
                </div>
                {prompt.tools.length === 0 ? (
                  <Text className="text-gray-500 text-xs">No tools added</Text>
                ) : (
                  <div className="space-y-2">
                    {prompt.tools.map((tool, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-2 bg-gray-50 border border-gray-200 rounded"
                      >
                        <div className="flex-1 min-w-0">
                          <div className="font-medium text-xs truncate">{tool.name}</div>
                          <div className="text-xs text-gray-500 truncate">{tool.description}</div>
                        </div>
                        <div className="flex items-center space-x-1 ml-2">
                          <button
                            onClick={() => openToolModal(index)}
                            className="text-xs text-blue-600 hover:text-blue-700"
                          >
                            Edit
                          </button>
                          <button
                            onClick={() => removeTool(index)}
                            className="text-gray-400 hover:text-red-500"
                          >
                            <TrashIcon size={14} />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </Card>

              {/* Developer Message Card */}
              <Card className="p-3">
                <Text className="block mb-2 text-sm font-medium">Developer message</Text>
                <Text className="text-gray-500 text-xs mb-2">Optional system instructions for the model</Text>
                <VariableTextArea
                  value={prompt.developerMessage}
                  onChange={(value) =>
                    setPrompt({
                      ...prompt,
                      developerMessage: value,
                    })
                  }
                  rows={3}
                  placeholder="e.g., You are a helpful assistant..."
                />
              </Card>

              {/* Prompt Messages Card */}
              <Card className="p-3">
                <div className="mb-2">
                  <Text className="text-sm font-medium">Prompt messages</Text>
                  <Text className="text-gray-500 text-xs mt-1">
                    Use <code className="bg-gray-100 px-1 rounded text-xs">{'{{variable}}'}</code> syntax for template variables
                  </Text>
                </div>
                <div className="space-y-2">
                  {prompt.messages.map((message, index) => (
                    <div key={index} className="border border-gray-300 rounded overflow-hidden bg-white">
                      <div className="bg-gray-50 px-2 py-1.5 border-b border-gray-300 flex items-center justify-between">
                        <Select
                          value={message.role}
                          onChange={(value) => updateMessage(index, "role", value)}
                          style={{ width: 100 }}
                          size="small"
                          bordered={false}
                        >
                          <Option value="user">User</Option>
                          <Option value="assistant">Assistant</Option>
                          <Option value="system">System</Option>
                        </Select>
                        {prompt.messages.length > 1 && (
                          <button
                            onClick={() => removeMessage(index)}
                            className="text-gray-400 hover:text-red-500"
                          >
                            <TrashIcon size={14} />
                          </button>
                        )}
                      </div>
                      <div className="p-2">
                        <VariableTextArea
                          value={message.content}
                          onChange={(value) => updateMessage(index, "content", value)}
                          rows={3}
                          placeholder="Enter prompt content..."
                        />
                      </div>
                    </div>
                  ))}
                </div>
                <button
                  onClick={addMessage}
                  className="mt-2 text-xs text-blue-600 hover:text-blue-700 flex items-center"
                >
                  <PlusIcon size={14} className="mr-1" />
                  Add message
                </button>
              </Card>
            </div>
          </div>

          {/* Right Panel - Preview/Conversation */}
          <div className="flex-1 bg-white flex flex-col">
            <div className="flex-1 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <div className="w-12 h-12 mx-auto mb-3 bg-gray-100 rounded-full flex items-center justify-center">
                  <MessageSquareIcon size={24} className="text-gray-400" />
                </div>
                <p className="text-sm">Your conversation will appear here</p>
                <p className="text-xs text-gray-500 mt-2">Save the prompt to test it</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Name Modal */}
      <Modal
        title="Publish Prompt"
        open={showNameModal}
        onCancel={() => setShowNameModal(false)}
        footer={[
          <div key="footer" className="flex justify-end gap-2">
            <TremorButton variant="secondary" onClick={() => setShowNameModal(false)}>
              Cancel
            </TremorButton>
            <TremorButton onClick={handleSave} loading={isSaving}>
              Publish
            </TremorButton>
          </div>
        ]}
      >
        <div className="py-4">
          <Text className="mb-2">Name</Text>
          <Input
            value={prompt.name}
            onChange={(e) => setPrompt({ ...prompt, name: e.target.value })}
            placeholder="Enter prompt name"
            onPressEnter={handleSave}
            autoFocus
          />
          <Text className="text-gray-500 text-xs mt-2">
            Published prompts can be used in API calls and are versioned for easy tracking.
          </Text>
        </div>
      </Modal>

      {/* Tool Modal */}
      {showToolModal && (
        <ToolModal
          visible={showToolModal}
          initialJson={editingToolIndex !== null ? prompt.tools[editingToolIndex].json : ""}
          onSave={addTool}
          onClose={() => {
            setShowToolModal(false);
            setEditingToolIndex(null);
          }}
        />
      )}
    </div>
  );
};

export default PromptEditorView;

