import React, { useState } from "react";
import ToolModal from "../tool_modal";
import NotificationsManager from "../../molecules/notifications_manager";
import { createPromptCall } from "../../networking";
import { PromptType, PromptEditorViewProps, Tool } from "./types";
import { extractVariables, convertToDotPrompt } from "./utils";
import PromptEditorHeader from "./PromptEditorHeader";
import ModelConfigCard from "./ModelConfigCard";
import ToolsCard from "./ToolsCard";
import DeveloperMessageCard from "./DeveloperMessageCard";
import PromptMessagesCard from "./PromptMessagesCard";
import ConversationPanel from "./ConversationPanel";
import PublishModal from "./PublishModal";

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

  const moveMessage = (fromIndex: number, toIndex: number) => {
    const newMessages = [...prompt.messages];
    const [movedMessage] = newMessages.splice(fromIndex, 1);
    newMessages.splice(toIndex, 0, movedMessage);
    setPrompt({
      ...prompt,
      messages: newMessages,
    });
  };

  const addTool = (json: string) => {
    try {
      const parsed = JSON.parse(json);
      const tool: Tool = {
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
    if (!prompt.name || prompt.name.trim() === "" || prompt.name === "New prompt") {
      setShowNameModal(true);
    } else {
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
      const variables = extractVariables(prompt);

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

  return (
    <div className="flex h-full bg-white">
      <div className="flex-1 flex flex-col">
        <PromptEditorHeader
          promptName={prompt.name}
          onNameChange={(name) => setPrompt({ ...prompt, name })}
          onBack={onClose}
          onSave={handleSaveClick}
          isSaving={isSaving}
        />

        <div className="flex-1 flex overflow-hidden">
          <div className="w-1/2 overflow-y-auto bg-white border-r border-gray-200">
            <div className="p-6 space-y-4 pb-20">
              <ModelConfigCard
                model={prompt.model}
                temperature={prompt.config.temperature}
                maxTokens={prompt.config.max_tokens}
                accessToken={accessToken}
                onModelChange={(model) => setPrompt({ ...prompt, model })}
                onTemperatureChange={(temperature) =>
                  setPrompt({
                    ...prompt,
                    config: { ...prompt.config, temperature },
                  })
                }
                onMaxTokensChange={(max_tokens) =>
                  setPrompt({
                    ...prompt,
                    config: { ...prompt.config, max_tokens },
                  })
                }
              />

              <ToolsCard
                tools={prompt.tools}
                onAddTool={() => openToolModal()}
                onEditTool={openToolModal}
                onRemoveTool={removeTool}
              />

              <DeveloperMessageCard
                value={prompt.developerMessage}
                onChange={(developerMessage) => setPrompt({ ...prompt, developerMessage })}
              />

              <PromptMessagesCard
                messages={prompt.messages}
                onAddMessage={addMessage}
                onUpdateMessage={updateMessage}
                onRemoveMessage={removeMessage}
                onMoveMessage={moveMessage}
              />
            </div>
          </div>

          <ConversationPanel />
        </div>
      </div>

      <PublishModal
        visible={showNameModal}
        promptName={prompt.name}
        isSaving={isSaving}
        onNameChange={(name) => setPrompt({ ...prompt, name })}
        onPublish={handleSave}
        onCancel={() => setShowNameModal(false)}
      />

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

