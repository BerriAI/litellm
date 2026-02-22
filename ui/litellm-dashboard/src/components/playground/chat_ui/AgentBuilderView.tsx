"use client";

import { CommentOutlined, DeleteOutlined, ExperimentOutlined, LinkOutlined, PlusOutlined, RobotOutlined, SaveOutlined } from "@ant-design/icons";
import { Button, Input, Modal, Select, Spin, Tabs } from "antd";
import React, { useCallback, useEffect, useState } from "react";
import CodeBlock from "@/app/(dashboard)/api-reference/components/CodeBlock";
import NotificationsManager from "../../molecules/notifications_manager";
import { keyCreateCall, modelCreateCall, modelDeleteCall, modelPatchUpdateCall, proxyBaseUrl } from "../../networking";
import { fetchMCPServers } from "../../networking";
import { MCPServer } from "../../mcp_tools/types";
import { AgentModel, fetchAvailableAgentModels, MCPToolEntry } from "../llm_calls/fetch_agents";
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

function getConnectTabBaseUrl(
  proxySettings: AgentBuilderViewProps["proxySettings"],
  customProxyBaseUrl?: string,
): string {
  const customDocBaseUrl = proxySettings?.LITELLM_UI_API_DOC_BASE_URL;
  if (customDocBaseUrl && customDocBaseUrl.trim()) return customDocBaseUrl;
  if (proxySettings?.PROXY_BASE_URL) return proxySettings.PROXY_BASE_URL;
  if (customProxyBaseUrl?.trim()) return customProxyBaseUrl;
  return "<your_proxy_base_url>";
}

interface ConnectTabContentProps {
  agentName: string;
  proxySettings: AgentBuilderViewProps["proxySettings"];
  customProxyBaseUrl?: string;
  accessToken: string | null;
  userID: string | null;
  disabledPersonalKeyCreation: boolean;
  creatingKey: boolean;
  createdKeyValue: string | null;
  onCreateKey: () => void;
}

function ConnectTabContent({
  agentName,
  proxySettings,
  customProxyBaseUrl,
  disabledPersonalKeyCreation,
  creatingKey,
  createdKeyValue,
  onCreateKey,
}: ConnectTabContentProps) {
  const baseUrl = proxyBaseUrl ?? getConnectTabBaseUrl(proxySettings, customProxyBaseUrl);
  const apiKeyForCurl =
    createdKeyValue ?
      createdKeyValue.startsWith("Bearer ") ? createdKeyValue : `Bearer ${createdKeyValue}`
    : "Bearer sk-1234";
  const curlExample = `curl -L -X POST '${baseUrl}/v1/chat/completions' \\
-H 'x-litellm-api-key: ${apiKeyForCurl}' \\
-d '{
  "model": "${agentName}",
  "stream": true,
  "stream_options": {
    "include_usage": true
  },
  "messages": [
    {
      "role": "user",
      "content": "hey"
    }
  ]
}'`;
  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h3 className="text-sm font-semibold text-gray-900 mb-1">Proxy base URL</h3>
        <p className="text-sm text-gray-600 font-mono bg-gray-50 px-2 py-1.5 rounded border border-gray-200 break-all">
          {baseUrl}
        </p>
      </div>
      <div>
        <h3 className="text-sm font-semibold text-gray-900 mb-2">Call your agent (cURL)</h3>
        <CodeBlock code={curlExample} language="bash" />
      </div>
      <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
        <h3 className="text-sm font-semibold text-gray-900 mb-2">Create a key for this agent</h3>
        <p className="text-sm text-gray-600 mb-3">
          Create a virtual key that can only call this agent. The key will be scoped to you (user_id) and restricted to
          the model <span className="font-mono text-gray-800">{agentName}</span>.
        </p>
        <Button
          type="primary"
          onClick={onCreateKey}
          loading={creatingKey}
          disabled={disabledPersonalKeyCreation}
        >
          Create key for this agent
        </Button>
        {disabledPersonalKeyCreation && (
          <p className="text-xs text-amber-600 mt-2">Key creation is disabled for your account.</p>
        )}
        {createdKeyValue && (
          <p className="text-xs text-green-700 mt-2">
            Key created. It is shown in the cURL example above — copy the snippet to use it.
          </p>
        )}
      </div>
    </div>
  );
}

function getAgentModelId(agent: AgentModel): string | null {
  const info = agent.model_info as { id?: string } | null | undefined;
  return info?.id ?? null;
}

function parseUnderlyingModel(litellmModel: string | undefined): string | undefined {
  if (!litellmModel || !litellmModel.startsWith("litellm_agent/")) return undefined;
  return litellmModel.slice("litellm_agent/".length) || undefined;
}

const MCP_TOOLS_PREFIX = "litellm_proxy/mcp/";

function buildToolsFromServerIds(serverIds: string[], servers: MCPServer[]): MCPToolEntry[] {
  return serverIds.map((serverId) => {
    const server = servers.find((s) => s.server_id === serverId);
    const serverName = server?.alias || server?.server_name || serverId;
    return {
      type: "mcp",
      server_label: "litellm",
      server_url: `${MCP_TOOLS_PREFIX}${serverName}`,
      require_approval: "never",
    };
  });
}

function getServerIdsFromTools(tools: MCPToolEntry[], servers: MCPServer[]): string[] {
  return tools
    .filter((t) => t.type === "mcp" && t.server_url?.startsWith(MCP_TOOLS_PREFIX))
    .map((t) => {
      const suffix = t.server_url.slice(MCP_TOOLS_PREFIX.length);
      const server = servers.find((s) => (s.alias || s.server_name || s.server_id) === suffix);
      return server?.server_id;
    })
    .filter((id): id is string => id != null);
}

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
  const [activeTab, setActiveTab] = useState<"configure" | "chat" | "test" | "connect">("configure");
  const [creatingKey, setCreatingKey] = useState(false);
  const [createdKeyValue, setCreatedKeyValue] = useState<string | null>(null);

  // Draft for new agent
  const [draftName, setDraftName] = useState("");
  const [draftSystemPrompt, setDraftSystemPrompt] = useState("");
  const [draftUnderlyingModel, setDraftUnderlyingModel] = useState<string | undefined>(undefined);
  const [draftTemperature, setDraftTemperature] = useState(0.7);
  const [draftMaxTokens, setDraftMaxTokens] = useState(4096);
  const [draftTools, setDraftTools] = useState<MCPToolEntry[]>([]);

  const [mcpServers, setMCPServers] = useState<MCPServer[]>([]);
  const [loadingMCPServers, setLoadingMCPServers] = useState(false);

  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const effectiveApiKey = apiKey || accessToken || "";
  const selectedAgent = selectedId === NEW_AGENT_ID ? null : agentModels.find((a) => a.model_name === selectedId) ?? null;
  const isNewAgent = selectedId === NEW_AGENT_ID;
  const selectedAgentModelId = selectedAgent ? getAgentModelId(selectedAgent) : null;

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

  const loadMCPServers = useCallback(async () => {
    if (!effectiveApiKey) return;
    setLoadingMCPServers(true);
    try {
      const servers = await fetchMCPServers(effectiveApiKey);
      setMCPServers(Array.isArray(servers) ? servers : (servers as { data?: MCPServer[] })?.data ?? []);
    } catch (e) {
      console.error("Error fetching MCP servers:", e);
    } finally {
      setLoadingMCPServers(false);
    }
  }, [effectiveApiKey]);

  useEffect(() => {
    loadMCPServers();
  }, [loadMCPServers]);

  // Clear created key when switching to another agent
  useEffect(() => {
    setCreatedKeyValue(null);
  }, [selectedId]);

  // Sync draft fields when selecting an existing agent
  useEffect(() => {
    if (selectedAgent && !isNewAgent) {
      setDraftName(selectedAgent.model_name);
      setDraftSystemPrompt(selectedAgent.litellm_params?.litellm_system_prompt ?? "");
      const underlying = parseUnderlyingModel(selectedAgent.litellm_params?.model);
      setDraftUnderlyingModel(underlying ?? modelGroups[0]?.model_group);
      const p = selectedAgent.litellm_params as { temperature?: number; max_tokens?: number } | undefined;
      setDraftTemperature(typeof p?.temperature === "number" ? p.temperature : 0.7);
      setDraftMaxTokens(typeof p?.max_tokens === "number" ? p.max_tokens : 4096);
      const rawTools = selectedAgent.litellm_params?.tools;
      const tools: MCPToolEntry[] = Array.isArray(rawTools)
        ? rawTools.filter((t): t is MCPToolEntry => t && typeof t === "object" && (t as MCPToolEntry).type === "mcp" && typeof (t as MCPToolEntry).server_url === "string")
        : [];
      setDraftTools(tools);
    }
  }, [selectedId, isNewAgent, selectedAgent?.model_name, selectedAgent?.litellm_params?.tools]);

  const selectedMCPServerIds = getServerIdsFromTools(draftTools, mcpServers);

  const handleMCPServerChange = (serverIds: string[]) => {
    setDraftTools(buildToolsFromServerIds(serverIds, mcpServers));
  };

  const handleAddAgent = () => {
    setSelectedId(NEW_AGENT_ID);
    setDraftName("");
    setDraftSystemPrompt("You are a helpful assistant.");
    setDraftUnderlyingModel(modelGroups[0]?.model_group);
    setDraftTemperature(0.7);
    setDraftMaxTokens(4096);
    setDraftTools([]);
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
          tools: draftTools,
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

  const handleUpdateAgent = async () => {
    if (!accessToken || !selectedAgent || !selectedAgentModelId || !draftName?.trim() || !draftUnderlyingModel) {
      NotificationsManager.fromBackend("Name and underlying model are required");
      return;
    }
    setSaving(true);
    try {
      await modelPatchUpdateCall(
        accessToken,
        {
          model_name: draftName.trim(),
          litellm_params: {
            model: `litellm_agent/${draftUnderlyingModel}`,
            litellm_system_prompt: draftSystemPrompt.trim() || undefined,
            temperature: draftTemperature,
            max_tokens: draftMaxTokens,
            tools: draftTools,
          },
          model_info: selectedAgent.model_info ?? {},
        },
        selectedAgentModelId,
      );
      NotificationsManager.success("Agent updated successfully");
      await loadAgents();
      setSelectedId(draftName.trim());
    } catch (e) {
      NotificationsManager.fromBackend("Failed to update agent");
    } finally {
      setSaving(false);
    }
  };

  const handleCreateKeyForAgent = async () => {
    if (!accessToken || !userID || !selectedAgent) return;
    setCreatingKey(true);
    setCreatedKeyValue(null);
    try {
      const response = await keyCreateCall(accessToken, userID, {
        models: [selectedAgent.model_name],
        key_alias: `Agent: ${selectedAgent.model_name}`,
      });
      const keyValue = response?.key ?? null;
      if (keyValue) {
        setCreatedKeyValue(keyValue);
        NotificationsManager.success("Virtual key created. Use it in the curl example below.");
      } else {
        NotificationsManager.fromBackend("Key created but value not returned");
      }
    } catch (e) {
      NotificationsManager.fromBackend("Failed to create key for agent");
    } finally {
      setCreatingKey(false);
    }
  };

  const handleDeleteAgent = () => {
    if (!selectedAgent || !selectedAgentModelId || !accessToken) return;
    Modal.confirm({
      title: "Delete agent",
      content: `Are you sure you want to delete "${selectedAgent.model_name}"? This cannot be undone.`,
      okText: "Delete",
      okType: "danger",
      cancelText: "Cancel",
      onOk: async () => {
        setDeleting(true);
        try {
          await modelDeleteCall(accessToken, selectedAgentModelId);
          NotificationsManager.success("Agent deleted");
          await loadAgents();
          const remaining = agentModels.filter((a) => a.model_name !== selectedAgent.model_name);
          setSelectedId(remaining.length > 0 ? remaining[0].model_name : null);
        } catch (e) {
          NotificationsManager.fromBackend("Failed to delete agent");
        } finally {
          setDeleting(false);
        }
      },
    });
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
            Agent Builder is experimental and may change or be removed without notice. We’d love your feedback—email us at{" "}
            <a href="mailto:product@berri.ai" className="font-medium text-amber-900 underline hover:text-amber-700">
              product@berri.ai
            </a>
            .
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
                onChange={(k) => setActiveTab(k as "configure" | "chat" | "test" | "connect")}
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
                        {(isNewAgent || selectedAgent) ? (
                          <div className="mx-auto max-w-xl space-y-4">
                            {!selectedAgentModelId && selectedAgent && (
                              <div className="rounded border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
                                This agent cannot be updated or deleted here (missing model id). Manage it from Models &amp; Endpoints.
                              </div>
                            )}
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
                            <div>
                              <label className="mb-1 block text-sm font-medium text-gray-700">MCP servers</label>
                              <Select
                                mode="multiple"
                                placeholder="Select MCP servers to attach (same format as chat completions API)"
                                value={selectedMCPServerIds}
                                onChange={handleMCPServerChange}
                                loading={loadingMCPServers}
                                className="w-full"
                                allowClear
                                showSearch
                                optionFilterProp="label"
                                options={mcpServers.map((s) => ({
                                  value: s.server_id,
                                  label: s.alias || s.server_name || s.server_id,
                                }))}
                              />
                              {selectedAgent && draftTools.length > 0 && (
                                <p className="mt-1 text-xs text-gray-500">
                                  {draftTools.length} MCP server{draftTools.length !== 1 ? "s" : ""} saved. Use the same <code className="rounded bg-gray-100 px-1">tools</code> array in chat completions when calling this agent.
                                </p>
                              )}
                            </div>
                            {selectedAgent && (
                              <div className="flex flex-wrap items-center gap-2 pt-2">
                                {selectedAgentModelId && (
                                  <>
                                    <Button
                                      type="primary"
                                      icon={<SaveOutlined />}
                                      onClick={handleUpdateAgent}
                                      loading={saving}
                                      disabled={!draftName?.trim() || !draftUnderlyingModel}
                                    >
                                      Update Agent
                                    </Button>
                                    <Button
                                      type="default"
                                      danger
                                      icon={<DeleteOutlined />}
                                      onClick={handleDeleteAgent}
                                      loading={deleting}
                                    >
                                      Delete
                                    </Button>
                                  </>
                                )}
                                <Button type="primary" icon={<CommentOutlined />} onClick={() => setActiveTab("chat")}>
                                  Test in Chat
                                </Button>
                              </div>
                            )}
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
                  {
                    key: "connect",
                    label: (
                      <span>
                        <LinkOutlined className="mr-1" /> Connect
                      </span>
                    ),
                    disabled: isNewAgent,
                    children: (
                      <div className="h-full overflow-y-auto p-6">
                        {selectedAgent ? (
                          <ConnectTabContent
                            agentName={selectedAgent.model_name}
                            proxySettings={proxySettings}
                            customProxyBaseUrl={customProxyBaseUrl}
                            accessToken={accessToken}
                            userID={userID}
                            disabledPersonalKeyCreation={disabledPersonalKeyCreation}
                            creatingKey={creatingKey}
                            createdKeyValue={createdKeyValue}
                            onCreateKey={handleCreateKeyForAgent}
                          />
                        ) : (
                          <div className="flex flex-1 items-center justify-center text-gray-500">
                            Select an agent to see how to connect.
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
