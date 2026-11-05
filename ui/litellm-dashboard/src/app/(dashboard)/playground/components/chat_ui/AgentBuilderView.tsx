"use client";

import { Bot, FlaskConical, Link as LinkIcon, MessageSquare, Plus, Save, Trash2 } from "lucide-react";
import React, { useCallback, useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { useVisitedTabs } from "@/hooks/useVisitedTabs";
import CodeBlock from "@/components/CodeBlock";
import { toast } from "@/lib/toast";
import {
  keyCreateCall,
  modelCreateCall,
  modelDeleteCall,
  modelPatchUpdateCall,
  proxyBaseUrl,
} from "@/components/networking";
import { fetchMCPServers } from "@/components/networking";
import { MCPServer } from "@/components/mcp_tools/types";
import { AgentModel, fetchAvailableAgentModels, MCPToolEntry } from "../../llm_calls/fetch_agents";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import ComplianceUI from "../complianceUI/ComplianceUI";
import ChatUI from "./ChatUI";

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

type AgentTab = "configure" | "chat" | "test" | "connect";

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
  const { t } = useTranslation();
  const baseUrl = proxyBaseUrl ?? getConnectTabBaseUrl(proxySettings, customProxyBaseUrl);
  const apiKeyForCurl = createdKeyValue
    ? createdKeyValue.startsWith("Bearer ")
      ? createdKeyValue
      : `Bearer ${createdKeyValue}`
    : "Bearer <your-master-key>";
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
        <h3 className="text-sm font-semibold text-foreground mb-1">
          {t("playground.chatUi.agentBuilderView.proxyBaseUrl")}
        </h3>
        <p className="text-sm text-muted-foreground font-mono bg-muted px-2 py-1.5 rounded-sm border border-border break-all">
          {baseUrl}
        </p>
      </div>
      <div>
        <h3 className="text-sm font-semibold text-foreground mb-2">
          {t("playground.chatUi.agentBuilderView.callYourAgent")}
        </h3>
        <CodeBlock code={curlExample} language="bash" />
      </div>
      <div className="rounded-lg border border-border bg-muted p-4">
        <h3 className="text-sm font-semibold text-foreground mb-2">
          {t("playground.chatUi.agentBuilderView.createKeyTitle")}
        </h3>
        <p className="text-sm text-muted-foreground mb-3">
          <Trans
            i18nKey="playground.chatUi.agentBuilderView.createKeyDescription"
            components={{ model: <span className="font-mono text-foreground">{agentName}</span> }}
          />
        </p>
        <Button onClick={onCreateKey} disabled={creatingKey || disabledPersonalKeyCreation}>
          {t("playground.chatUi.agentBuilderView.createKeyButton")}
        </Button>
        {disabledPersonalKeyCreation && (
          <p className="text-xs text-warning mt-2">{t("playground.chatUi.agentBuilderView.keyCreationDisabled")}</p>
        )}
        {createdKeyValue && (
          <p className="text-xs text-success mt-2">{t("playground.chatUi.agentBuilderView.keyCreated")}</p>
        )}
      </div>
    </div>
  );
}

function getAgentModelId(agent: AgentModel): string | null {
  const info = agent.model_info as { id?: string } | null | undefined;
  return info?.id ?? null;
}

// Selection key that always resolves to a non-null string. Prefers the DB
// id (stable across renames and unique across teams) but falls back to
// `model_name` so config-file-defined agents — which have no `model_info.id`
// — remain selectable.
function getAgentSelectionKey(agent: AgentModel): string {
  return getAgentModelId(agent) ?? agent.model_name;
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
  const { t } = useTranslation();
  const [agentModels, setAgentModels] = useState<AgentModel[]>([]);
  const [modelGroups, setModelGroups] = useState<ModelGroup[]>([]);
  const [loadingAgents, setLoadingAgents] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<AgentTab>("configure");
  const { onTabChange, hasVisited } = useVisitedTabs("configure");
  const goToTab = (tab: AgentTab) => {
    setActiveTab(tab);
    onTabChange(tab);
  };
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
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  const effectiveApiKey = apiKey || accessToken || "";
  const selectedAgent =
    selectedId === NEW_AGENT_ID ? null : agentModels.find((a) => getAgentSelectionKey(a) === selectedId) ?? null;
  const isNewAgent = selectedId === NEW_AGENT_ID;
  const selectedAgentModelId = selectedAgent ? getAgentModelId(selectedAgent) : null;

  const loadAgents = useCallback(async (): Promise<AgentModel[]> => {
    if (!accessToken || !userID || !userRole) return [];
    setLoadingAgents(true);
    try {
      const list = await fetchAvailableAgentModels(accessToken, userID, userRole);
      setAgentModels(list);
      if (!selectedId || (selectedId !== NEW_AGENT_ID && !list.some((a) => getAgentSelectionKey(a) === selectedId))) {
        setSelectedId(list.length > 0 ? getAgentSelectionKey(list[0]) : null);
      }
      return list;
    } catch (e) {
      console.error(e);
      toast.fromError(t("playground.chatUi.agentBuilderView.loadAgentsFailed"));
      return [];
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
        ? rawTools.filter(
            (t): t is MCPToolEntry =>
              t &&
              typeof t === "object" &&
              (t as MCPToolEntry).type === "mcp" &&
              typeof (t as MCPToolEntry).server_url === "string",
          )
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
    goToTab("configure");
  };

  const handleSaveAgent = async () => {
    if (!accessToken || !draftName?.trim() || !draftUnderlyingModel) {
      toast.fromError(t("playground.chatUi.agentBuilderView.nameAndModelRequired"));
      return;
    }
    setSaving(true);
    try {
      const response = await modelCreateCall(accessToken, {
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
      // /model/new returns the row with `model_id` at the top level.
      // Prefer that id over name-matching so we land on the just-created
      // agent even when its public name collides with another team's.
      const createdId: string | null = response?.model_id ?? response?.model_info?.id ?? null;
      const list = await loadAgents();
      const created = createdId
        ? list.find((a) => getAgentModelId(a) === createdId) ?? list.find((a) => a.model_name === draftName.trim())
        : list.find((a) => a.model_name === draftName.trim());
      setSelectedId(created ? getAgentSelectionKey(created) : list[0] ? getAgentSelectionKey(list[0]) : null);
      goToTab("chat");
    } catch (e) {
      toast.fromError(t("playground.chatUi.agentBuilderView.saveFailed"));
    } finally {
      setSaving(false);
    }
  };

  const handleUpdateAgent = async () => {
    if (!accessToken || !selectedAgent || !selectedAgentModelId || !draftName?.trim() || !draftUnderlyingModel) {
      toast.fromError(t("playground.chatUi.agentBuilderView.nameAndModelRequired"));
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
      toast.success(t("playground.chatUi.agentBuilderView.updated"));
      const list = await loadAgents();
      const stillSelected = list.find((a) => getAgentModelId(a) === selectedAgentModelId);
      const target = stillSelected ?? list[0];
      setSelectedId(target ? getAgentSelectionKey(target) : null);
    } catch (e) {
      toast.fromError(t("playground.chatUi.agentBuilderView.updateFailed"));
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
        toast.success(t("playground.chatUi.agentBuilderView.virtualKeyCreated"));
      } else {
        toast.fromError(t("playground.chatUi.agentBuilderView.keyValueMissing"));
      }
    } catch (e) {
      toast.fromError(t("playground.chatUi.agentBuilderView.createKeyFailed"));
    } finally {
      setCreatingKey(false);
    }
  };

  const handleDeleteAgent = () => {
    if (!selectedAgent || !selectedAgentModelId || !accessToken) return;
    setConfirmingDelete(true);
  };

  const handleConfirmDelete = async () => {
    if (!selectedAgent || !selectedAgentModelId || !accessToken) return;
    setDeleting(true);
    try {
      await modelDeleteCall(accessToken, selectedAgentModelId);
      toast.success(t("playground.chatUi.agentBuilderView.deleted"));
      const list = await loadAgents();
      const remaining = list.filter((a) => getAgentModelId(a) !== selectedAgentModelId);
      setSelectedId(remaining.length > 0 ? getAgentSelectionKey(remaining[0]) : null);
    } catch (e) {
      toast.fromError(t("playground.chatUi.agentBuilderView.deleteFailed"));
    } finally {
      setDeleting(false);
      setConfirmingDelete(false);
    }
  };

  if (!accessToken || !userID || !userRole) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-muted-foreground">
        {t("playground.chatUi.agentBuilderView.signIn")}
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col bg-card text-foreground">
      <div className="flex shrink-0 flex-col border-b border-border">
        <div className="flex h-12 items-center justify-between px-4">
          <span className="text-sm font-medium text-foreground">{t("playground.chatUi.agentBuilderView.title")}</span>
          {isNewAgent ? (
            <Button onClick={handleSaveAgent} disabled={saving || !draftName?.trim() || !draftUnderlyingModel}>
              <Save />
              {t("playground.chatUi.agentBuilderView.saveAgent")}
            </Button>
          ) : (
            <span className="text-xs text-muted-foreground">{t("playground.chatUi.agentBuilderView.subtitle")}</span>
          )}
        </div>
        <div className="flex items-center gap-2 border-t border-warning/20 bg-warning/10 px-4 py-2 text-xs text-warning">
          <FlaskConical className="size-4 shrink-0 text-warning" />
          <span>
            <Trans
              i18nKey="playground.chatUi.agentBuilderView.experimentalNotice"
              components={{
                email: (
                  <a
                    href="mailto:product@berri.ai"
                    className="font-medium text-warning underline hover:text-warning/80"
                  />
                ),
              }}
            />
          </span>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Roster */}
        <div className="w-60 shrink-0 border-r border-border bg-card flex flex-col">
          <div className="flex items-center justify-between border-b border-border p-3">
            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {t("playground.chatUi.agentBuilderView.agents")}
            </span>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={handleAddAgent}
              aria-label={t("playground.chatUi.agentBuilderView.addAgent")}
            >
              <Plus />
            </Button>
          </div>
          <div className="flex-1 overflow-y-auto p-2">
            {loadingAgents ? (
              <div className="flex justify-center py-4" aria-busy="true">
                <UiLoadingSpinner className="size-4 text-muted-foreground" />
              </div>
            ) : (
              <>
                {agentModels.map((agent) => {
                  const key = getAgentSelectionKey(agent);
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => setSelectedId(key)}
                      className={`mb-1 w-full rounded-md border-l-2 px-3 py-2 text-left text-sm transition-colors ${
                        selectedId === key ? "border-info bg-info/10 text-info" : "border-transparent hover:bg-accent"
                      }`}
                    >
                      <div className="font-medium truncate">{agent.model_name}</div>
                      <div className="text-[10px] text-muted-foreground truncate">litellm_agent</div>
                    </button>
                  );
                })}
                <button
                  type="button"
                  onClick={handleAddAgent}
                  className="mb-1 w-full rounded-md border border-dashed border-border px-3 py-2 text-left text-sm text-muted-foreground hover:border-info hover:bg-info/10 hover:text-foreground"
                >
                  <Plus className="mr-1 inline size-4" /> {t("playground.chatUi.agentBuilderView.newAgent")}
                </button>
              </>
            )}
          </div>
        </div>

        {/* Main content */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {selectedId === null && !isNewAgent && agentModels.length === 0 && !loadingAgents && (
            <div className="flex flex-1 items-center justify-center p-8 text-muted-foreground">
              {t("playground.chatUi.agentBuilderView.noAgents")}
            </div>
          )}
          {(selectedId !== null || isNewAgent) && (
            <>
              <Tabs
                value={activeTab}
                onValueChange={(value) => goToTab(value as AgentTab)}
                className="flex flex-1 flex-col overflow-hidden"
              >
                <TabsList variant="line" className="h-auto w-full justify-start rounded-none border-b p-0 pl-4">
                  <TabsTrigger value="configure" className="flex-none rounded-none px-4 py-2">
                    <Bot />
                    {t("playground.chatUi.agentBuilderView.tabs.configure")}
                  </TabsTrigger>
                  <TabsTrigger value="chat" disabled={isNewAgent} className="flex-none rounded-none px-4 py-2">
                    <MessageSquare />
                    {t("playground.chatUi.agentBuilderView.tabs.chat")}
                  </TabsTrigger>
                  <TabsTrigger value="test" disabled={isNewAgent} className="flex-none rounded-none px-4 py-2">
                    <FlaskConical />
                    {t("playground.chatUi.agentBuilderView.tabs.batchTest")}
                  </TabsTrigger>
                  <TabsTrigger value="connect" disabled={isNewAgent} className="flex-none rounded-none px-4 py-2">
                    <LinkIcon />
                    {t("playground.chatUi.agentBuilderView.tabs.connect")}
                  </TabsTrigger>
                </TabsList>

                <TabsContent
                  value="configure"
                  keepMounted={hasVisited("configure")}
                  className="min-h-0 overflow-hidden"
                >
                  <div className="h-full overflow-y-auto p-6">
                    {isNewAgent || selectedAgent ? (
                      <div className="mx-auto max-w-xl space-y-4">
                        {!selectedAgentModelId && selectedAgent && (
                          <div className="rounded-sm border border-warning/20 bg-warning/10 px-3 py-2 text-xs text-warning">
                            {t("playground.chatUi.agentBuilderView.missingModelId")}
                          </div>
                        )}
                        <div>
                          <label className="mb-1 block text-sm font-medium text-foreground">
                            {t("playground.chatUi.agentBuilderView.agentName")}
                          </label>
                          <Input
                            value={draftName}
                            onChange={(e) => setDraftName(e.target.value)}
                            placeholder={t("playground.chatUi.agentBuilderView.agentNamePlaceholder")}
                          />
                        </div>
                        <div>
                          <label className="mb-1 block text-sm font-medium text-foreground">
                            {t("playground.chatUi.agentBuilderView.systemPrompt")}
                          </label>
                          <Textarea
                            value={draftSystemPrompt}
                            onChange={(e) => setDraftSystemPrompt(e.target.value)}
                            placeholder={t("playground.chatUi.agentBuilderView.systemPromptPlaceholder")}
                            rows={6}
                            className="field-sizing-fixed"
                          />
                        </div>
                        <div>
                          <label className="mb-1 block text-sm font-medium text-foreground">
                            {t("playground.chatUi.agentBuilderView.underlyingLlm")}
                          </label>
                          <Select
                            value={draftUnderlyingModel ?? null}
                            onValueChange={(model: string | null) => setDraftUnderlyingModel(model ?? undefined)}
                          >
                            <SelectTrigger
                              className="w-full"
                              aria-label={t("playground.chatUi.agentBuilderView.underlyingLlm")}
                            >
                              <SelectValue placeholder={t("playground.chatUi.agentBuilderView.selectModel")} />
                            </SelectTrigger>
                            <SelectContent>
                              {modelGroups.map((m) => (
                                <SelectItem key={m.model_group} value={m.model_group}>
                                  {m.model_group}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <label className="mb-1 block text-sm font-medium text-foreground">
                              {t("playground.chatUi.agentBuilderView.temperature")}
                            </label>
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
                            <label className="mb-1 block text-sm font-medium text-foreground">
                              {t("playground.chatUi.agentBuilderView.maxTokens")}
                            </label>
                            <Input
                              type="number"
                              min={1}
                              value={draftMaxTokens}
                              onChange={(e) => setDraftMaxTokens(Number(e.target.value))}
                            />
                          </div>
                        </div>
                        <div>
                          <label className="mb-1 block text-sm font-medium text-foreground">
                            {t("playground.chatUi.agentBuilderView.mcpServers")}
                          </label>
                          <MultiSelect
                            placeholder={t("playground.chatUi.agentBuilderView.mcpServersPlaceholder")}
                            value={selectedMCPServerIds}
                            onValueChange={handleMCPServerChange}
                            loading={loadingMCPServers}
                            className="w-full"
                            options={mcpServers.map((s) => ({
                              value: s.server_id,
                              label: s.alias || s.server_name || s.server_id,
                            }))}
                          />
                          {selectedAgent && draftTools.length > 0 && (
                            <p className="mt-1 text-xs text-muted-foreground">
                              <Trans
                                i18nKey="playground.chatUi.agentBuilderView.mcpServersSaved"
                                count={draftTools.length}
                                components={{ field: <code className="rounded-sm bg-muted px-1" /> }}
                              />
                            </p>
                          )}
                        </div>
                        {selectedAgent && (
                          <div className="flex flex-wrap items-center gap-2 pt-2">
                            {selectedAgentModelId && (
                              <>
                                <Button
                                  onClick={handleUpdateAgent}
                                  disabled={saving || !draftName?.trim() || !draftUnderlyingModel}
                                >
                                  <Save />
                                  {t("playground.chatUi.agentBuilderView.updateAgent")}
                                </Button>
                                <Button variant="destructive" onClick={handleDeleteAgent} disabled={deleting}>
                                  <Trash2 />
                                  {t("playground.chatUi.agentBuilderView.delete")}
                                </Button>
                              </>
                            )}
                            <Button onClick={() => goToTab("chat")}>
                              <MessageSquare />
                              {t("playground.chatUi.agentBuilderView.testInChat")}
                            </Button>
                          </div>
                        )}
                      </div>
                    ) : null}
                  </div>
                </TabsContent>
                <TabsContent value="chat" keepMounted={hasVisited("chat")} className="min-h-0 overflow-hidden">
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
                      <div className="flex flex-1 items-center justify-center text-muted-foreground">
                        {t("playground.chatUi.agentBuilderView.saveFirst")}
                      </div>
                    )}
                  </div>
                </TabsContent>
                <TabsContent value="test" keepMounted={hasVisited("test")} className="min-h-0 overflow-hidden">
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
                      <div className="flex flex-1 items-center justify-center text-muted-foreground">
                        {t("playground.chatUi.agentBuilderView.selectForBatch")}
                      </div>
                    )}
                  </div>
                </TabsContent>
                <TabsContent value="connect" keepMounted={hasVisited("connect")} className="min-h-0 overflow-hidden">
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
                      <div className="flex flex-1 items-center justify-center text-muted-foreground">
                        {t("playground.chatUi.agentBuilderView.selectForConnect")}
                      </div>
                    )}
                  </div>
                </TabsContent>
              </Tabs>
            </>
          )}
        </div>
      </div>

      <AlertDialog open={confirmingDelete} onOpenChange={setConfirmingDelete}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("playground.chatUi.agentBuilderView.deleteDialog.title")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("playground.chatUi.agentBuilderView.deleteDialog.description", { name: selectedAgent?.model_name })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogAction variant="outline">
              {t("playground.chatUi.agentBuilderView.deleteDialog.cancel")}
            </AlertDialogAction>
            <Button variant="destructive" onClick={handleConfirmDelete} disabled={deleting}>
              {t("playground.chatUi.agentBuilderView.deleteDialog.confirm")}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
