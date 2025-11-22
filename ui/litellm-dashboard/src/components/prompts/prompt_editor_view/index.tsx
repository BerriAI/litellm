import React, { useState } from "react";
import ToolModal from "../tool_modal";
import NotificationsManager from "../../molecules/notifications_manager";
import { createPromptCall, updatePromptCall } from "../../networking";
import { PromptType, PromptEditorViewProps, Tool } from "./types";
import { convertToDotPrompt, parseExistingPrompt } from "./utils";
import PromptEditorHeader from "./PromptEditorHeader";
import ModelConfigCard from "./ModelConfigCard";
import ToolsCard from "./ToolsCard";
import DeveloperMessageCard from "./DeveloperMessageCard";
import PromptMessagesCard from "./PromptMessagesCard";
import ConversationPanel from "./conversation_panel";
import PublishModal from "./PublishModal";
import DotpromptViewTab from "./DotpromptViewTab";
import VersionHistorySidePanel from "./VersionHistorySidePanel";

const PromptEditorView: React.FC<PromptEditorViewProps> = ({ onClose, onSuccess, accessToken, initialPromptData }) => {
  const getInitialPrompt = (): PromptType => {
    if (initialPromptData) {
      try {
        return parseExistingPrompt(initialPromptData);
      } catch (error) {
        console.error("Error parsing existing prompt:", error);
        NotificationsManager.fromBackend("Failed to parse prompt data");
      }
    }
    return {
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
    };
  };

  const [prompt, setPrompt] = useState<PromptType>(getInitialPrompt());
  const [editMode, setEditMode] = useState<boolean>(!!initialPromptData);
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  
  // Construct versioned ID from prompt_id and version field
  const getInitialVersionId = () => {
    if (!initialPromptData?.prompt_spec) return undefined;
    const baseId = initialPromptData.prompt_spec.prompt_id;
    const version = initialPromptData.prompt_spec.version || 
                   (initialPromptData.prompt_spec.litellm_params as any)?.prompt_id;
    
    // If version is a number, construct versioned ID
    if (typeof version === 'number') {
      return `${baseId}.v${version}`;
    }
    
    // If version is a string with version suffix, use it
    if (typeof version === 'string' && (version.includes('.v') || version.includes('_v'))) {
      return version;
    }
    
    return baseId;
  };
  
  const [activeVersionId, setActiveVersionId] = useState<string | undefined>(getInitialVersionId());

  const [showToolModal, setShowToolModal] = useState(false);
  const [showNameModal, setShowNameModal] = useState(false);
  const [editingToolIndex, setEditingToolIndex] = useState<number | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [viewMode, setViewMode] = useState<"pretty" | "dotprompt">("pretty");

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

  const handleLoadVersion = (versionData: any) => {
    try {
      const loadedPrompt = parseExistingPrompt({ prompt_spec: versionData });
      setPrompt(loadedPrompt);
      // Store the version number or construct versioned ID for tracking
      const versionNum = versionData.version || 1;
      setActiveVersionId(`${versionData.prompt_id}.v${versionNum}`);
      // NotificationsManager.success(`Loaded version v${versionNum}`);
    } catch (error) {
      console.error("Error loading version:", error);
      NotificationsManager.fromBackend("Failed to load prompt version");
    }
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
      const promptId = prompt.name.replace(/[^a-zA-Z0-9_-]/g, "_").toLowerCase();
      const dotpromptContent = convertToDotPrompt(prompt);

      const promptData = {
        prompt_id: promptId,
        litellm_params: {
          prompt_integration: "dotprompt",
          prompt_id: promptId,
          dotprompt_content: dotpromptContent,
        },
        prompt_info: {
          prompt_type: "db",
        },
      };

      if (editMode && initialPromptData?.prompt_spec?.prompt_id) {
        await updatePromptCall(accessToken, initialPromptData.prompt_spec.prompt_id, promptData);
        NotificationsManager.success("Prompt updated successfully!");
      } else {
        await createPromptCall(accessToken, promptData);
        NotificationsManager.success("Prompt created successfully!");
      }
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Error saving prompt:", error);
      NotificationsManager.fromBackend(editMode ? "Failed to update prompt" : "Failed to save prompt");
    } finally {
      setIsSaving(false);
      setShowNameModal(false);
    }
  };

  const getVersionNumber = (pid?: string) => {
    if (!pid) return null;
    if (pid.includes(".v")) {
      return `v${pid.split(".v")[1]}`;
    }
    return null;
  };

  const currentVersion = getVersionNumber(activeVersionId);

  // Extract template variables from prompt content for code examples
  const extractTemplateVariables = (): Record<string, string> => {
    const variables: Record<string, string> = {};
    const allContent = [
      prompt.developerMessage,
      ...prompt.messages.map(m => m.content)
    ].join(' ');
    
    const variableRegex = /\{\{(\w+)\}\}/g;
    let match;
    while ((match = variableRegex.exec(allContent)) !== null) {
      const varName = match[1];
      if (!variables[varName]) {
        variables[varName] = `example_${varName}`;
      }
    }
    return variables;
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
          editMode={editMode}
          onShowHistory={() => setShowHistoryModal(true)}
          version={currentVersion}
          promptModel={prompt.model}
          promptVariables={extractTemplateVariables()}
          accessToken={accessToken}
        />

        <div className="flex-1 flex overflow-hidden">
          <div className="w-1/2 overflow-y-auto bg-white border-r border-gray-200 flex-shrink-0">
            <div className="border-b border-gray-200 bg-white px-6 py-4 flex items-center gap-3">
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

              <div className="ml-auto inline-flex items-center bg-gray-200 rounded-full p-0.5">
                <button
                  className={`px-3 py-1 text-xs font-medium rounded-full transition-colors ${
                    viewMode === "pretty" ? "bg-white text-gray-900 shadow-sm" : "text-gray-600"
                  }`}
                  onClick={() => setViewMode("pretty")}
                >
                  PRETTY
                </button>
                <button
                  className={`px-3 py-1 text-xs font-medium rounded-full transition-colors ${
                    viewMode === "dotprompt" ? "bg-white text-gray-900 shadow-sm" : "text-gray-600"
                  }`}
                  onClick={() => setViewMode("dotprompt")}
                >
                  DOTPROMPT
                </button>
              </div>
            </div>

            {viewMode === "pretty" ? (
              <div className="p-6 space-y-4 pb-20">
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
            ) : (
              <DotpromptViewTab prompt={prompt} />
            )}
          </div>

          <div className="w-1/2 flex-shrink-0">
            <ConversationPanel prompt={prompt} accessToken={accessToken} />
          </div>
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

      <VersionHistorySidePanel
        isOpen={showHistoryModal}
        onClose={() => setShowHistoryModal(false)}
        accessToken={accessToken}
        promptId={initialPromptData?.prompt_spec?.prompt_id || prompt.name}
        activeVersionId={activeVersionId}
        onSelectVersion={handleLoadVersion}
      />
    </div>
  );
};

export default PromptEditorView;
