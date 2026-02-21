"use client";

import { CommentOutlined, ExperimentOutlined, PlusOutlined, RobotOutlined, SaveOutlined } from "@ant-design/icons";
import { Button, Input, Select, Spin, Tabs } from "antd";
import React, { useCallback, useEffect, useState } from "react";
import NotificationsManager from "../../molecules/notifications_manager";
import { modelCreateCall } from "../../networking";
import { AgentModel, fetchAvailableAgentModels } from "../llm_calls/fetch_agents";
import { fetchAvailableModels, ModelGroup } from "../llm_calls/fetch_models";
import ComplianceUI from "../complianceUI/ComplianceUI";
import ChatUI from "./ChatUI";

const { TextArea } = Input;

export interface AgentBuilderViewProps {
  accessToken: string | null;
  token: string | null;
  userID: string | null;
  userRole: string | null;
  disabledPersonalKeyCreation?: boolean;
  proxySettings?: {
    PROXY_BASE_URL?: string;
    LITELLM_UI_API_DOC_BASE_URL?: string | null;
  };
  apiKey?: string;
  customProxyBaseUrl?: string;
}

const NEW_AGENT_ID = "__new__";

export default function AgentBuilderView({
  accessToken,
  token,
  userID,
  userRole,
  disabledPersonalKeyCreation = false,
  proxySettings,
  apiKey,
  customProxyBaseUrl,
}: AgentBuilderViewProps) {
  const [agentModels, setAgentModels] = useState<AgentModel[]>([]);
  const [modelGroups, setModelGroups] = useState<ModelGroup[]>([]);
  const [loadingAgents, setLoadingAgents] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"configure" | "chat" | "test">("configure");

  // Draft for new agent
  const [draftName, setDraftName] = useState("");
  const [draftSystemPrompt, setDraftSystemPrompt] = useState("");
  const [draftUnderlyingModel, setDraftUnderlyingModel] = useState<string | undefined>(undefined);
  const [draftTemperature, setDraftTemperature] = useState(0.7);
  const [draftMaxTokens, setDraftMaxTokens] = useState(4096);

  const [saving, setSaving] = useState(false);

  const effectiveApiKey = apiKey || accessToken || "";
  const selectedAgent = selectedId === NEW_AGENT_ID ? null : agentModels.find((a) => a.model_name === selectedId) ?? null;
  const isNewAgent = selectedId === NEW_AGENT_ID;

  const loadAgents = useCallback(async () => {
    if (!accessToken || !userID || !userRole) return;
    setLoadingAgents(true);
    try {
      const list = await fetchAvailableAgentModels(accessToken, userID, userRole);
      setAgentModels(list);
      if (!selectedId || (selectedId !== NEW_AGENT_ID && !list.some((a) => a.model_name === selectedId))) {
        setSelectedId(list.length > 0 ? list[0].model_name : null);
      }
    } catch (e) {
      console.error(e);
      NotificationsManager.fromBackend("Failed to load agents");
    } finally {
      setLoadingAgents(false);
    }
  }, [accessToken, userID, userRole]);

  const loadModels = useCallback(async () => {
    if (!effectiveApiKey) return;
    try {
      const models = await fetchAvailableModels(effectiveApiKey);
      setModelGroups(models);
      if (!draftUnderlyingModel && models.length > 0) {
        setDraftUnderlyingModel(models[0].model_group);
      }
    } catch (e) {
      console.error(e);
    }
  }, [effectiveApiKey]);

  useEffect(() => {
    loadAgents();
  }, [loadAgents]);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const handleAddAgent = () => {
    setSelectedId(NEW_AGENT_ID);
    setDraftName("");
    setDraftSystemPrompt("You are a helpful assistant.");
    setDraftUnderlyingModel(modelGroups[0]?.model_group);
    setDraftTemperature(0.7);
    setDraftMaxTokens(4096);
    setActiveTab("configure");
  };

  const handleSaveAgent = async () => {
    if (!accessToken || !draftName?.trim() || !draftUnderlyingModel) {
      NotificationsManager.fromBackend("Name and underlying model are required");
      return;
    }
    setSaving(true);
    try {
      await modelCreateCall(accessToken, {
        model_name: draftName.trim(),
        litellm_params: {
          model: `litellm_agent/${draftUnderlyingModel}`,
          litellm_system_prompt: draftSystemPrompt.trim() || undefined,
          temperature: draftTemperature,
          max_tokens: draftMaxTokens,
        },
        model_info: {},
      });
      const newName = draftName.trim();
      await loadAgents();
      setSelectedId(newName);
      setActiveTab("chat");
    } catch (e) {
      NotificationsManager.fromBackend("Failed to save agent");
    } finally {
      setSaving(false);
    }
  };

  if (!accessToken || !userID || !userRole) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-gray-500">
        Sign in to use Agent Builder.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col bg-white text-gray-900">
      <div className="flex flex-shrink-0 flex-col border-b border-gray-200">
        <div className="flex h-12 items-center justify-between px-4">
          <span className="text-sm font-medium text-gray-900">Agent Builder</span>
        {isNewAgent ? (
          <Button
            type="primary"
            icon={<SaveOutlined />}
            onClick={handleSaveAgent}
            loading={saving}
            disabled={!draftName?.trim() || !draftUnderlyingModel}
          >
            Save Agent
          </Button>
        ) : (
          <span className="text-xs text-gray-500">Build Agents that pass your compliance requirements.</span>
        )}
        </div>
        <div className="flex items-center gap-2 border-t border-amber-200 bg-amber-50 px-4 py-2 text-xs text-amber-800">
          <ExperimentOutlined className="flex-shrink-0 text-amber-600" />
          <span>
            Agent Builder is experimental and may change or be removed without notice.
          </span>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Roster */}
        <div className="w-60 flex-shrink-0 border-r border-gray-200 bg-white flex flex-col">
          <div className="flex items-center justify-between border-b border-gray-200 p-3">
            <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">Agents</span>
            <Button type="text" size="small" icon={<PlusOutlined />} onClick={handleAddAgent} aria-label="Add agent" />
          </div>
          <div className="flex-1 overflow-y-auto p-2">
            {loadingAgents ? (
              <div className="flex justify-center py-4">
                <Spin size="small" />
              </div>
            ) : (
              <>
                {agentModels.map((agent) => (
                  <button
                    key={agent.model_name}
                    type="button"
                    onClick={() => setSelectedId(agent.model_name)}
                    className={`mb-1 w-full rounded-md border-l-2 px-3 py-2 text-left text-sm transition-colors ${
                      selectedId === agent.model_name
                        ? "border-blue-500 bg-blue-50 text-blue-800"
                        : "border-transparent hover:bg-gray-50"
                    }`}
                  >
                    <div className="font-medium truncate">{agent.model_name}</div>
                    <div className="text-[10px] text-gray-500 truncate">litellm_agent</div>
                  </button>
                ))}
                <button
                  type="button"
                  onClick={handleAddAgent}
                  className="mb-1 w-full rounded-md border border-dashed border-gray-300 px-3 py-2 text-left text-sm text-gray-500 hover:border-blue-400 hover:bg-blue-50/50 hover:text-gray-700"
                >
                  <PlusOutlined className="mr-1" /> New agent
                </button>
              </>
            )}
          </div>
        </div>

        {/* Main content */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {selectedId === null && !isNewAgent && agentModels.length === 0 && !loadingAgents && (
            <div className="flex flex-1 items-center justify-center p-8 text-gray-500">
              No agents yet. Add an agent to get started.
            </div>
          )}
          {(selectedId !== null || isNewAgent) && (
            <>
              <Tabs
                activeKey={activeTab}
                onChange={(k) => setActiveTab(k as "configure" | "chat" | "test")}
                className="flex-1 overflow-hidden [&_.ant-tabs-content]:h-full [&_.ant-tabs-tabpane]:h-full [&_.ant-tabs-nav]:pl-4"
                items={[
                  {
                    key: "configure",
                    label: (
                      <span>
                        <RobotOutlined className="mr-1" /> Configure
                      </span>
                    ),
                    children: (
                      <div className="h-full overflow-y-auto p-6">
                        {isNewAgent ? (
                          <div className="mx-auto max-w-xl space-y-4">
                            <div>
                              <label className="mb-1 block text-sm font-medium text-gray-700">Agent name</label>
                              <Input
                                value={draftName}
                                onChange={(e) => setDraftName(e.target.value)}
                                placeholder="My Agent"
                              />
                            </div>
                            <div>
                              <label className="mb-1 block text-sm font-medium text-gray-700">System prompt</label>
                              <TextArea
                                value={draftSystemPrompt}
                                onChange={(e) => setDraftSystemPrompt(e.target.value)}
                                placeholder="You are a helpful assistant..."
                                rows={6}
                              />
                            </div>
                            <div>
                              <label className="mb-1 block text-sm font-medium text-gray-700">Underlying LLM</label>
                              <Select
                                value={draftUnderlyingModel}
                                onChange={setDraftUnderlyingModel}
                                className="w-full"
                                options={modelGroups.map((m) => ({ value: m.model_group, label: m.model_group }))}
                                placeholder="Select model"
                              />
                            </div>
                            <div className="grid grid-cols-2 gap-4">
                              <div>
                                <label className="mb-1 block text-sm font-medium text-gray-700">Temperature</label>
                                <Input
                                  type="number"
                                  min={0}
                                  max={2}
                                  step={0.1}
                                  value={draftTemperature}
                                  onChange={(e) => setDraftTemperature(Number(e.target.value))}
                                />
                              </div>
                              <div>
                                <label className="mb-1 block text-sm font-medium text-gray-700">Max tokens</label>
                                <Input
                                  type="number"
                                  min={1}
                                  value={draftMaxTokens}
                                  onChange={(e) => setDraftMaxTokens(Number(e.target.value))}
                                />
                              </div>
                            </div>
                          </div>
                        ) : selectedAgent ? (
                          <div className="mx-auto max-w-xl space-y-4">
                            <div>
                              <label className="mb-1 block text-sm font-medium text-gray-700">Name</label>
                              <div className="rounded border border-gray-200 bg-gray-50 px-3 py-2 text-sm">
                                {selectedAgent.model_name}
                              </div>
                            </div>
                            <div>
                              <label className="mb-1 block text-sm font-medium text-gray-700">System prompt</label>
                              <div className="rounded border border-gray-200 bg-gray-50 px-3 py-2 text-sm whitespace-pre-wrap">
                                {selectedAgent.litellm_params?.litellm_system_prompt || "(none)"}
                              </div>
                            </div>
                            <div>
                              <label className="mb-1 block text-sm font-medium text-gray-700">Underlying model</label>
                              <div className="rounded border border-gray-200 bg-gray-50 px-3 py-2 text-sm font-mono">
                                {selectedAgent.litellm_params?.model ?? ""}
                              </div>
                            </div>
                            <Button type="primary" icon={<CommentOutlined />} onClick={() => setActiveTab("chat")}>
                              Test in Chat
                            </Button>
                          </div>
                        ) : null}
                      </div>
                    ),
                  },
                  {
                    key: "chat",
                    label: (
                      <span>
                        <CommentOutlined className="mr-1" /> Chat
                      </span>
                    ),
                    disabled: isNewAgent,
                    children: (
                      <div className="flex h-full flex-col min-h-0">
                        {selectedAgent ? (
                          <ChatUI
                            key={selectedAgent.model_name}
                            simplified
                            fixedModel={selectedAgent.model_name}
                            accessToken={accessToken}
                            token={token}
                            userRole={userRole}
                            userID={userID}
                            disabledPersonalKeyCreation={disabledPersonalKeyCreation}
                            proxySettings={proxySettings}
                          />
                        ) : (
                          <div className="flex flex-1 items-center justify-center text-gray-500">
                            Save an agent first to test in Chat.
                          </div>
                        )}
                      </div>
                    ),
                  },
                  {
                    key: "test",
                    label: (
                      <span>
                        <ExperimentOutlined className="mr-1" /> Batch Test
                      </span>
                    ),
                    disabled: isNewAgent,
                    children: (
                      <div className="flex h-full flex-col min-h-0">
                        {selectedAgent ? (
                          <ComplianceUI
                            accessToken={accessToken}
                            disabledPersonalKeyCreation={disabledPersonalKeyCreation}
                            backendMode="chat_completions"
                            fixedModel={selectedAgent.model_name}
                            proxySettings={proxySettings}
                          />
                        ) : (
                          <div className="flex flex-1 items-center justify-center text-gray-500">
                            Select an agent to run batch tests.
                          </div>
                        )}
                      </div>
                    ),
                  },
                ]}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
