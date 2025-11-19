import React, { useState, useEffect } from "react";
import { Button as TremorButton } from "@tremor/react";
import { Input, Select } from "antd";
import {
  PlusIcon,
  MoreHorizontalIcon,
  FlaskConicalIcon,
  SaveIcon,
  MessageSquareIcon,
  SettingsIcon,
  TrashIcon,
  ArrowLeftIcon,
} from "lucide-react";
import ToolModal from "./tool_modal";
import ModelSelector from "../common_components/ModelSelector";
import NotificationsManager from "../molecules/notifications_manager";
import { createPromptCall } from "../networking";

const { TextArea } = Input;
const { Option } = Select;

interface Message {
  role: string;
  content: string;
}

interface Variable {
  name: string;
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
  variables: Variable[];
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
    variables: [],
    tools: [],
    developerMessage: "",
    messages: [
      {
        role: "user",
        content: "Enter task specifics. Use {{template variables}} for dynamic inputs",
      },
    ],
  });

  const [showConfig, setShowConfig] = useState(false);
  const [showToolModal, setShowToolModal] = useState(false);
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

  const addVariable = () => {
    setPrompt({
      ...prompt,
      variables: [...prompt.variables, { name: "" }],
    });
  };

  const updateVariable = (index: number, value: string) => {
    const newVariables = [...prompt.variables];
    newVariables[index].name = value;
    setPrompt({
      ...prompt,
      variables: newVariables,
    });
  };

  const removeVariable = (index: number) => {
    setPrompt({
      ...prompt,
      variables: prompt.variables.filter((_, i) => i !== index),
    });
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

  const handleSave = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("Access token is required");
      return;
    }

    if (!prompt.name || prompt.name.trim() === "" || prompt.name === "New prompt") {
      NotificationsManager.fromBackend("Please enter a valid prompt name");
      return;
    }

    setIsSaving(true);
    try {
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
              schema: prompt.variables.reduce((acc, v) => {
                if (v.name) {
                  acc[v.name] = "string";
                }
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
    }
  };

  const convertToDotPrompt = (prompt: PromptType): string => {
    let result = `---\nmodel: ${prompt.model}\n\ninput:\n  schema:\n`;

    prompt.variables.forEach((variable) => {
      if (variable.name) {
        result += `    ${variable.name}: string\n`;
      }
    });

    result += `\noutput:\n  format: text\n---\n\n`;

    prompt.messages.forEach((message) => {
      result += `${message.role}: ${message.content}\n\n`;
    });

    return result;
  };

  return (
    <div className="flex h-full bg-gray-50">
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
              onClick={handleSave}
              loading={isSaving}
              disabled={isSaving}
            >
              Save
            </TremorButton>
          </div>
        </div>

        <div className="flex-1 flex overflow-hidden">
          {/* Left Panel - Editor */}
          <div className="flex-1 overflow-y-auto">
            <div className="max-w-3xl mx-auto p-6 space-y-6">
              {/* Model Selector */}
              <div>
                <ModelSelector
                  accessToken={accessToken || ""}
                  value={prompt.model}
                  onChange={(model) =>
                    setPrompt({
                      ...prompt,
                      model,
                    })
                  }
                  showLabel={true}
                  labelText="Model"
                />
                <button
                  onClick={() => setShowConfig(!showConfig)}
                  className="mt-3 flex items-center text-sm font-medium text-gray-700 hover:text-gray-900"
                >
                  <SettingsIcon size={16} className="mr-1" />
                  <span>Configuration</span>
                </button>
                {showConfig && (
                  <div className="mt-3 p-3 bg-gray-50 rounded text-xs space-y-2">
                    <div>
                      <label className="text-gray-600 block mb-1">Temperature</label>
                      <Input
                        type="number"
                        value={prompt.config.temperature}
                        onChange={(e) =>
                          setPrompt({
                            ...prompt,
                            config: {
                              ...prompt.config,
                              temperature: parseFloat(e.target.value),
                            },
                          })
                        }
                        step="0.1"
                        min="0"
                        max="2"
                      />
                    </div>
                    <div>
                      <label className="text-gray-600 block mb-1">Max Tokens</label>
                      <Input
                        type="number"
                        value={prompt.config.max_tokens}
                        onChange={(e) =>
                          setPrompt({
                            ...prompt,
                            config: {
                              ...prompt.config,
                              max_tokens: parseInt(e.target.value),
                            },
                          })
                        }
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Variables */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700">Variables</label>
                  <button
                    onClick={addVariable}
                    className="text-sm text-gray-600 hover:text-gray-900 flex items-center"
                  >
                    <PlusIcon size={16} className="mr-1" />
                    Add
                  </button>
                </div>
                {prompt.variables.length > 0 && (
                  <div className="space-y-2">
                    {prompt.variables.map((variable, index) => (
                      <div key={index} className="flex items-center space-x-2">
                        <Input
                          value={variable.name}
                          onChange={(e) => updateVariable(index, e.target.value)}
                          placeholder="e.g. city"
                        />
                        <button
                          onClick={() => removeVariable(index)}
                          className="p-2 text-gray-400 hover:text-red-500"
                        >
                          <TrashIcon size={18} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Tools */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700">Tools</label>
                  <button
                    onClick={() => openToolModal()}
                    className="text-sm text-gray-600 hover:text-gray-900 flex items-center"
                  >
                    <PlusIcon size={16} className="mr-1" />
                    Add
                  </button>
                </div>
                {prompt.tools.length > 0 && (
                  <div className="space-y-2">
                    {prompt.tools.map((tool, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-3 bg-gray-50 border border-gray-200 rounded"
                      >
                        <div className="flex-1">
                          <div className="font-medium text-sm">{tool.name}</div>
                          <div className="text-xs text-gray-500">{tool.description}</div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <button
                            onClick={() => openToolModal(index)}
                            className="text-sm text-blue-600 hover:text-blue-700"
                          >
                            Edit
                          </button>
                          <button
                            onClick={() => removeTool(index)}
                            className="text-gray-400 hover:text-red-500"
                          >
                            <TrashIcon size={16} />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Developer Message */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium text-gray-700">Developer message</label>
                </div>
                <TextArea
                  value={prompt.developerMessage}
                  onChange={(e) =>
                    setPrompt({
                      ...prompt,
                      developerMessage: e.target.value,
                    })
                  }
                  rows={3}
                  placeholder="Optional system message"
                />
              </div>

              {/* Prompt Messages */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Prompt messages</label>
                <div className="space-y-3">
                  {prompt.messages.map((message, index) => (
                    <div key={index} className="border border-gray-300 rounded-lg overflow-hidden">
                      <div className="bg-gray-50 px-3 py-2 border-b border-gray-300 flex items-center justify-between">
                        <Select
                          value={message.role}
                          onChange={(value) => updateMessage(index, "role", value)}
                          style={{ width: 120 }}
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
                            <TrashIcon size={16} />
                          </button>
                        )}
                      </div>
                      <TextArea
                        value={message.content}
                        onChange={(e) => updateMessage(index, "content", e.target.value)}
                        rows={3}
                        placeholder="Enter task specifics. Use {{template variables}} for dynamic inputs"
                        bordered={false}
                        className="w-full"
                      />
                    </div>
                  ))}
                </div>
                <button
                  onClick={addMessage}
                  className="mt-3 text-sm text-gray-600 hover:text-gray-900 flex items-center"
                >
                  <PlusIcon size={16} className="mr-1" />
                  Add message
                </button>
              </div>
            </div>
          </div>

          {/* Right Panel - Preview/Conversation */}
          <div className="w-1/2 border-l border-gray-200 bg-white flex flex-col">
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

