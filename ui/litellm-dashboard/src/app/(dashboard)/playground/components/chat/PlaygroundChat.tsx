"use client";

import { useRef, useState } from "react";
import { ChevronDown, PanelRight, RotateCcw, SquarePen } from "lucide-react";
import useCan from "@/app/(dashboard)/hooks/useCan";
import { generateCodeSnippet } from "@/components/chat_ui/CodeSnippets";
import { EndpointType, getEndpointType } from "@/components/chat_ui/mode_endpoint_mapping";
import type { MessageType } from "@/components/chat_ui/types";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";
import { customHeadersFromPairs } from "@/components/llm_calls/request_headers";
import type { MCPToolArgumentsFormRef } from "@/components/mcp_tools/MCPToolArgumentsForm";
import type { MCPServer, MCPTool, MCPToolset } from "@/components/mcp_tools/types";
import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { toast } from "@/lib/toast";
import { useChatHistory } from "../../hooks/useChatHistory";
import { useCodeInterpreter } from "../../hooks/useCodeInterpreter";
import { filterModelsForEndpoint, isModelCompatibleWithEndpoint } from "../chat_ui/EndpointUtils";
import RealtimePlayground from "../chat_ui/RealtimePlayground";
import { Composer } from "./Composer";
import { EndpointMenu, VoiceMenu } from "./ComposerControls";
import { Conversation, formatCost } from "./Conversation";
import { capabilitiesFor, CODE_INTERPRETER_SUGGESTIONS, SUGGESTIONS } from "./endpointCapabilities";
import { InspectorPanel, type CodeSdk } from "./Inspector";
import { ByokKeyButton } from "./ByokConnect";
import { McpDialog, serverName } from "./McpDialog";
import { TargetPicker, type OptionGroup } from "./Pickers";
import { TOOLSET_PREFIX, withEndpoint, withReachableMcp } from "./playgroundSettings";
import { SettingsPopover } from "./SettingsPopover";
import {
  useGuardrailOptions,
  useMcpCatalog,
  usePlaygroundAgents,
  usePlaygroundModels,
  usePolicyOptions,
  useServerTools,
  useTagOptions,
  useVectorStoreOptions,
  type PickerOption,
  type ServerTools,
} from "./usePlaygroundCatalog";
import { usePlaygroundSend, type SendDeps } from "./usePlaygroundSend";
import { usePlaygroundSettings } from "./usePlaygroundSettings";
import { useIsWide } from "./useIsWide";

export type ProxySettings = Readonly<{ PROXY_BASE_URL?: string; LITELLM_UI_API_DOC_BASE_URL?: string | null }>;

const POLICY_PREFIX = "policy:";

const providerOf = (model: ModelGroup): string => model.providers?.[0] ?? "other";

const modelGroups = (models: readonly ModelGroup[]): readonly OptionGroup[] =>
  [...new Set(models.map(providerOf))].sort().map((provider) => ({
    label: provider,
    options: models
      .filter((model) => providerOf(model) === provider)
      .map((model) => ({ value: model.model_group, label: model.model_group })),
  }));

const modelEmptyText = (failed: boolean, hasKey: boolean): string => {
  if (failed) return "Unable to load models for this key";
  if (!hasKey) return "Enter a virtual key to load models";
  return "No models available for this endpoint";
};

const reachableMcp = (servers: readonly MCPServer[], toolsets: readonly MCPToolset[]): ReadonlySet<string> =>
  new Set([
    ...servers.map((server) => server.server_id),
    ...toolsets.map((toolset) => `${TOOLSET_PREFIX}${toolset.toolset_id}`),
  ]);

const toolsetFor = (selection: string | null, toolsets: readonly MCPToolset[]): MCPToolset | null =>
  toolsets.find((toolset) => `${TOOLSET_PREFIX}${toolset.toolset_id}` === selection) ?? null;

const backingServers = (selection: string | null, toolset: MCPToolset | null): readonly string[] => {
  if (toolset) return [...new Set(toolset.tools.map((tool) => tool.server_id))];
  return selection && !selection.startsWith(TOOLSET_PREFIX) ? [selection] : [];
};

type DirectTarget = Readonly<{ server: string; tool: MCPTool }>;

const toolsetTargets = (toolset: MCPToolset, loaded: Readonly<Record<string, ServerTools>>): DirectTarget[] =>
  toolset.tools
    .flatMap((entry) => {
      const tool = loaded[entry.server_id]?.tools.find((candidate) => candidate.name === entry.tool_name);
      return tool ? [{ server: entry.server_id, tool }] : [];
    })
    .filter((target, index, targets) => targets.findIndex((other) => other.tool.name === target.tool.name) === index);

const directTargets = (
  servers: readonly string[],
  toolset: MCPToolset | null,
  loaded: Readonly<Record<string, ServerTools>>,
): readonly DirectTarget[] => {
  if (toolset) return toolsetTargets(toolset, loaded);
  return servers.flatMap((server) => (loaded[server]?.tools ?? []).map((tool) => ({ server, tool })));
};

const prefixed = (options: readonly PickerOption[]): readonly PickerOption[] =>
  options.map((option) => ({ ...option, value: `${POLICY_PREFIX}${option.value}` }));

const lastTracedReply = (messages: readonly MessageType[]): number =>
  messages.findLastIndex((message) => message.role === "assistant" && message.trace !== undefined);

const totalCost = (messages: readonly MessageType[]): number =>
  messages.reduce((sum, message) => sum + (message.usage?.cost ?? message.trace?.cost ?? 0), 0);

const conversationSummary = (messages: readonly MessageType[]): string => {
  const requests = messages.filter((message) => message.role === "user").length;
  if (requests === 0) return "New chat";
  const label = requests === 1 ? "1 request" : `${requests} requests`;
  const cost = totalCost(messages);
  return cost > 0 ? `${label}, ${formatCost(cost)}` : label;
};

export function PlaygroundChat({
  accessToken,
  disabledPersonalKeyCreation,
  proxySettings,
}: {
  accessToken: string | null;
  disabledPersonalKeyCreation: boolean;
  proxySettings: ProxySettings | undefined;
}) {
  const { settings: stored, update, reset } = usePlaygroundSettings(disabledPersonalKeyCreation);
  const canViewPolicies = useCan("viewPolicies");
  const apiKey = (stored.keySource === "session" ? accessToken : stored.virtualKey.trim()) || null;
  const mcp = useMcpCatalog(apiKey);
  const settings = withReachableMcp(stored, mcp.isReady ? reachableMcp(mcp.servers, mcp.toolsets) : null);
  const capabilities = capabilitiesFor(settings.endpoint);
  const chat = useChatHistory({ simplified: false });
  const codeInterpreter = useCodeInterpreter();
  const toolArgsForm = useRef<MCPToolArgumentsFormRef | null>(null);
  const [inspected, setInspected] = useState<number | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);
  const isWide = useIsWide();

  const models = usePlaygroundModels(apiKey);
  const [checkedModels, setCheckedModels] = useState<readonly ModelGroup[] | undefined>(undefined);
  if (models.data && models.data !== checkedModels) {
    setCheckedModels(models.data);
    const loaded = models.data;
    update((current) =>
      current.model && !loaded.some((option) => option.model_group === current.model)
        ? { ...current, model: null }
        : current,
    );
  }
  const agents = usePlaygroundAgents(apiKey, settings.proxyBaseUrl, settings.endpoint === EndpointType.A2A_AGENTS);
  const guardrails = useGuardrailOptions(accessToken);
  const policies = usePolicyOptions(accessToken, canViewPolicies);
  const vectorStores = useVectorStoreOptions(accessToken);
  const tags = useTagOptions(accessToken);

  const directSelection = settings.endpoint === EndpointType.MCP ? settings.mcpSelection[0] ?? null : null;
  const directToolset = toolsetFor(directSelection, mcp.toolsets);
  const directServers = backingServers(directSelection, directToolset);
  const directServerTools = useServerTools(apiKey, directServers);
  const directByok = directToolset ? undefined : mcp.servers.find((server) => server.server_id === directSelection);
  const targets = directTargets(directServers, directToolset, directServerTools);
  const directTarget = targets.find((target) => target.tool.name === settings.directTool) ?? null;
  const directTool = directTarget?.tool ?? null;
  const toolsFailed = directServers.some((server) => directServerTools[server]?.failed);
  const customHeaders = customHeadersFromPairs(settings.headerPairs);

  const sendDeps: SendDeps = {
    settings,
    capabilities,
    apiKey,
    signedIn: Boolean(accessToken),
    customHeaders,
    mcpServers: mcp.servers,
    mcpToolsets: mcp.toolsets,
    directCall: () => (directTarget ? { serverId: directTarget.server, tool: directTarget.tool.name } : null),
    toolArgsForm,
    chat,
    codeInterpreter,
  };
  const draft = usePlaygroundSend(sendDeps);

  const messages = chat.chatHistory;
  const inspectedIndex = inspected ?? lastTracedReply(messages);
  const inspectedMessage = inspectedIndex >= 0 ? messages[inspectedIndex] ?? null : null;
  const modelList = models.data ?? [];

  const selectModel = (model: string) => {
    const info = modelList.find((option) => option.model_group === model);
    const switchTo =
      info?.mode && !isModelCompatibleWithEndpoint(info, settings.endpoint) ? getEndpointType(info.mode) : null;
    update((current) => ({ ...(switchTo ? withEndpoint(current, switchTo) : current), model }));
  };

  const inspect = (index: number) => {
    setInspected(index);
    if (!isWide) setSheetOpen(true);
  };

  const clearChat = () => {
    chat.clearChatHistory();
    draft.clearDraft();
    codeInterpreter.clearResult();
    setInspected(null);
  };

  const newChat = () => {
    clearChat();
    toast.success("Chat history cleared.");
  };

  const resetAll = () => {
    clearChat();
    reset();
    toast.success("Chat and settings reset");
  };

  const code = (sdk: CodeSdk): string => {
    const snippet: Parameters<typeof generateCodeSnippet>[0] = {
      apiKeySource: "custom",
      accessToken: null,
      apiKey: "",
      inputMessage: draft.text,
      chatHistory: messages,
      selectedTags: [...settings.tags],
      selectedVectorStores: [...settings.vectorStores],
      selectedGuardrails: [...settings.guardrails],
      selectedPolicies: [...settings.policies],
      selectedMCPServers: [...settings.mcpSelection],
      mcpServers: [...mcp.servers],
      mcpServerToolRestrictions: Object.fromEntries(
        Object.entries(settings.toolLimits).map(([server, tools]) => [server, [...tools]]),
      ),
      endpointType: settings.endpoint,
      selectedModel: settings.model ?? undefined,
      selectedSdk: sdk,
      selectedVoice: settings.voice,
      proxySettings,
      customHeaders,
    };
    return generateCodeSnippet(snippet);
  };

  const targetRow = (
    <>
      {capabilities.target === "model" && (
        <TargetPicker
          label="Model"
          placeholder="Select a model"
          groups={modelGroups(filterModelsForEndpoint([...modelList], settings.endpoint))}
          value={settings.model}
          onValueChange={selectModel}
          isLoading={models.isLoading}
          emptyText={modelEmptyText(models.isError, Boolean(apiKey))}
          allowCustom
        />
      )}
      {capabilities.target === "agent" && (
        <TargetPicker
          label="Agent"
          placeholder="Select an agent"
          groups={[
            {
              label: "Agents",
              options: (agents.data ?? []).map((agent) => ({
                value: agent.agent_name,
                label: agent.agent_name || agent.agent_id,
                description: agent.agent_card_params?.description,
              })),
            },
          ]}
          value={settings.agent}
          onValueChange={(agent) => update({ agent })}
          isLoading={agents.isLoading}
          emptyText="No agents available for this key"
        />
      )}
      {capabilities.target === "mcpTool" && (
        <>
          <TargetPicker
            label="Server"
            placeholder="Select an MCP server"
            groups={[
              {
                label: "MCP servers",
                options: mcp.servers.map((server) => ({ value: server.server_id, label: serverName(server) })),
              },
              {
                label: "Toolsets",
                options: mcp.toolsets.map((toolset) => ({
                  value: `${TOOLSET_PREFIX}${toolset.toolset_id}`,
                  label: toolset.toolset_name,
                  description: toolset.description,
                })),
              },
            ]}
            value={directSelection}
            onValueChange={(server) => update({ mcpSelection: [server], directTool: null })}
            isLoading={mcp.isLoading}
            emptyText="No MCP servers available for this key"
          />
          {directByok?.is_byok && (
            <ByokKeyButton server={directByok} name={serverName(directByok)} onConnected={mcp.refreshCredentials} />
          )}
          {directSelection && (
            <TargetPicker
              label="Tool"
              placeholder="Select a tool"
              groups={[
                {
                  label: "Tools",
                  options: targets.map(({ tool }) => ({
                    value: tool.name,
                    label: tool.name,
                    description: tool.description,
                  })),
                },
              ]}
              value={settings.directTool}
              onValueChange={(tool) => update({ directTool: tool })}
              isLoading={directServers.some((server) => directServerTools[server]?.isLoading)}
              emptyText={toolsFailed ? "Unable to load tools for this server" : "This server has no tools"}
            />
          )}
        </>
      )}
      <EndpointMenu
        value={settings.endpoint}
        onValueChange={(endpoint) => update((current) => withEndpoint(current, endpoint))}
      />
      {capabilities.voice && <VoiceMenu value={settings.voice} onValueChange={(voice) => update({ voice })} />}
    </>
  );

  const guardrailGroups: readonly OptionGroup[] = [
    { label: "Guardrails", options: guardrails.data ?? [] },
    ...(canViewPolicies ? [{ label: "Policies", options: prefixed(policies.data ?? []) }] : []),
  ];

  const composer = (
    <Composer
      capabilities={capabilities}
      endpoint={settings.endpoint}
      targetRow={targetRow}
      draft={draft}
      directTool={directTool}
      toolArgsForm={toolArgsForm}
      mcpControl={
        <McpDialog
          requestKey={apiKey}
          servers={mcp.servers}
          toolsets={mcp.toolsets}
          isLoading={mcp.isLoading}
          selection={settings.mcpSelection}
          toolLimits={settings.toolLimits}
          onChange={(mcpSelection, toolLimits) => update({ mcpSelection, toolLimits })}
          onCredentialSaved={mcp.refreshCredentials}
        />
      }
      vectorStores={{
        groups: [{ label: "Vector stores", options: vectorStores.data ?? [] }],
        value: settings.vectorStores,
        onValueChange: (value) => update({ vectorStores: value }),
        isLoading: vectorStores.isLoading,
      }}
      guardrails={{
        groups: guardrailGroups,
        value: [...settings.guardrails, ...settings.policies.map((policy) => `${POLICY_PREFIX}${policy}`)],
        onValueChange: (value) =>
          update({
            guardrails: value.filter((item) => !item.startsWith(POLICY_PREFIX)),
            policies: value
              .filter((item) => item.startsWith(POLICY_PREFIX))
              .map((item) => item.slice(POLICY_PREFIX.length)),
          }),
        isLoading: guardrails.isLoading || policies.isLoading,
      }}
      codeInterpreter={{ enabled: codeInterpreter.enabled, onToggle: codeInterpreter.setEnabled }}
      settingsControl={
        <SettingsPopover
          settings={settings}
          update={update}
          capabilities={capabilities}
          keysDisabled={disabledPersonalKeyCreation}
          tagOptions={tags.data ?? []}
          suggestedBaseUrl={proxySettings?.LITELLM_UI_API_DOC_BASE_URL ?? proxySettings?.PROXY_BASE_URL ?? null}
          sessions={{
            enabled: chat.useApiSessionManagement,
            responseId: chat.responsesSessionId,
            onToggle: chat.handleToggleSessionManagement,
          }}
        />
      }
      sendDisabled={draft.isLoading}
    />
  );

  const suggestions =
    settings.endpoint === EndpointType.RESPONSES && codeInterpreter.enabled
      ? CODE_INTERPRETER_SUGGESTIONS
      : SUGGESTIONS[settings.endpoint] ?? [];

  const inspector = <InspectorPanel message={inspectedMessage} code={code} />;

  return (
    <div className="flex min-h-0 flex-1">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <div className="flex h-12 shrink-0 items-center gap-2 border-b border-border px-4">
          <span className="text-xs text-muted-foreground tabular-nums">{conversationSummary(messages)}</span>
          <span className="ml-auto flex items-center gap-2">
            {!isWide && (
              <Button variant="outline" size="sm" onClick={() => setSheetOpen(true)}>
                <PanelRight />
                Inspector
              </Button>
            )}
            <ButtonGroup>
              <Button variant="outline" size="sm" onClick={newChat}>
                <SquarePen />
                New chat
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger
                  render={<Button variant="outline" size="icon-sm" aria-label="More chat actions" />}
                >
                  <ChevronDown />
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-64">
                  <DropdownMenuItem onClick={newChat}>
                    <SquarePen />
                    New chat, keep settings
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={resetAll}>
                    <RotateCcw />
                    Reset chat and settings
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </ButtonGroup>
          </span>
        </div>
        {settings.endpoint === EndpointType.REALTIME ? (
          <div className="flex min-h-0 flex-1 flex-col">
            <RealtimePlayground
              accessToken={apiKey ?? ""}
              selectedModel={settings.model ?? ""}
              customProxyBaseUrl={settings.proxyBaseUrl || undefined}
              selectedGuardrails={settings.guardrails.length > 0 ? [...settings.guardrails] : undefined}
            />
          </div>
        ) : (
          <Conversation
            messages={messages}
            isLoading={draft.isLoading}
            waitingFor={messages.at(-1)?.model}
            extras={{
              endpoint: settings.endpoint,
              mcpEvents: chat.mcpEvents,
              codeInterpreter: codeInterpreter.result,
              accessToken: apiKey ?? "",
            }}
            inspectedIndex={inspectedIndex}
            onInspect={inspect}
            suggestions={suggestions}
            onPickSuggestion={draft.setText}
          />
        )}
        <div className="mx-auto w-full max-w-3xl px-6 pb-6">{composer}</div>
      </div>
      {isWide ? (
        <aside aria-label="Inspector" className="flex w-96 shrink-0 flex-col border-l border-border">
          {inspector}
        </aside>
      ) : (
        <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
          <SheetContent side="right" className="w-96 gap-0 p-0" showCloseButton={false}>
            <SheetTitle className="sr-only">Inspector</SheetTitle>
            {inspector}
          </SheetContent>
        </Sheet>
      )}
    </div>
  );
}
