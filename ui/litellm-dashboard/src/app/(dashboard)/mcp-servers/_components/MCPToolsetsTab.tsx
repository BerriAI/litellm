import React, { useState } from "react";
import { z } from "zod";
import { SortingState } from "@tanstack/react-table";
import { Inbox, Plus, X } from "lucide-react";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useQueryClient } from "@tanstack/react-query";
import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import {
  useMCPServerTools,
  type MCPServerToolEntry,
  type MCPServerToolsState,
} from "@/app/(dashboard)/hooks/mcpServers/useMCPServerTools";
import { DataTable } from "@/components/shared/DataTable";
import { createMCPToolset, updateMCPToolset, deleteMCPToolset, getGatewayBaseUrl } from "@/components/networking";
import { MCPServer, MCPToolset, MCPToolsetTool } from "@/components/mcp_tools/types";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";
import { displayToolName, getMCPToolsetTableColumns } from "./MCPToolsetTableColumns";
import {
  deselectTools,
  filterServerTools,
  isToolSelected,
  matchesServerName,
  selectTools,
  selectionState,
  toggleTool,
  type SelectionState,
} from "./toolsetSelection";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface MCPToolsetsTabProps {
  accessToken: string | null;
  userRole: string | null;
}

const toolsetSchema = z.object({
  toolset_name: z.string().min(1, "Please enter a toolset name"),
  description: z.string(),
});

type ToolsetFormValues = z.infer<typeof toolsetSchema>;

// Typing starts a fetch of every server's tools; wait for a pause so each keystroke does not.
const SEARCH_DEBOUNCE_WAIT_MS = 300;

const IDLE_SERVER_TOOLS: MCPServerToolsState = { tools: [], status: "idle" };

const checkboxStateProps = (state: SelectionState) => ({
  checked: state === "all",
  indeterminate: state === "some",
});

interface ServerView {
  server: MCPServer;
  serverName: string;
  state: MCPServerToolsState;
  visibleTools: readonly MCPServerToolEntry[];
}

const isPending = (state: MCPServerToolsState) => state.status === "idle" || state.status === "loading";

// While searching, a server stays listed until its tools load and turn out not to match.
const isHiddenBySearch = (
  search: string,
  serverNames: readonly string[],
  state: MCPServerToolsState,
  visibleTools: readonly MCPServerToolEntry[],
) => {
  if (!search.trim() || state.status !== "success") return false;
  return visibleTools.length === 0 && !matchesServerName(search, serverNames);
};

const buildServerViews = (
  servers: readonly MCPServer[],
  toolsByServer: ReadonlyMap<string, MCPServerToolsState>,
  search: string,
): ServerView[] =>
  servers.flatMap((server) => {
    const state = toolsByServer.get(server.server_id) ?? IDLE_SERVER_TOOLS;
    const names = [server.alias ?? "", server.server_name ?? ""];
    const visibleTools = filterServerTools(search, names, state.tools);
    if (isHiddenBySearch(search, names, state, visibleTools)) return [];
    return [{ server, serverName: server.alias || server.server_name || server.server_id, state, visibleTools }];
  });

const summarizeSearch = (serverViews: readonly ServerView[]) => ({
  matchingTools: serverViews.flatMap(({ server, visibleTools }) =>
    visibleTools.map((tool) => ({ server_id: server.server_id, tool_name: tool.name })),
  ),
  pendingServerCount: serverViews.filter(({ state }) => isPending(state)).length,
});

// Why a selected tool cannot be confirmed against its server. The tool stays selected and is
// still saved; the note only keeps it from looking lost.
const selectedToolIssue = (
  tool: MCPToolsetTool,
  serverNameById: ReadonlyMap<string, string>,
  toolsByServer: ReadonlyMap<string, MCPServerToolsState>,
): string | undefined => {
  if (!serverNameById.has(tool.server_id)) return "server not found";
  const state = toolsByServer.get(tool.server_id);
  if (state?.status === "error") return "server unreachable";
  if (state?.status === "success" && !state.tools.some((t) => t.name === tool.tool_name)) return "tool not found";
  return undefined;
};

interface ServerToolGroupProps {
  serverId: string;
  serverName: string;
  state: MCPServerToolsState;
  visibleTools: readonly MCPServerToolEntry[];
  expanded: boolean;
  searchActive: boolean;
  selectedTools: MCPToolsetTool[];
  onToggleExpanded: () => void;
  onToggleTool: (tool: MCPToolsetTool) => void;
  onSetToolsSelected: (tools: MCPToolsetTool[], selected: boolean) => void;
}

function ServerToolGroup({
  serverId,
  serverName,
  state,
  visibleTools,
  expanded,
  searchActive,
  selectedTools,
  onToggleExpanded,
  onToggleTool,
  onSetToolsSelected,
}: ServerToolGroupProps) {
  const selectedCount = selectedTools.filter((t) => t.server_id === serverId).length;
  const visibleRefs = visibleTools.map((tool) => ({ server_id: serverId, tool_name: tool.name }));
  const groupState = selectionState(selectedTools, visibleRefs);

  const renderBody = () => {
    if (isPending(state)) {
      return (
        <div className="flex justify-center py-3">
          <UiLoadingSpinner className="size-4" />
        </div>
      );
    }
    if (state.status === "error") {
      return <p className="text-xs text-destructive px-2 py-2">Couldn&apos;t load tools for this server.</p>;
    }
    if (visibleTools.length === 0) {
      return (
        <p className="text-xs text-muted-foreground px-2 py-2">
          {searchActive ? "No matching tools on this server." : "No tools found for this server."}
        </p>
      );
    }
    return (
      <div className="flex flex-col gap-1">
        {visibleRefs.map((ref, idx) => {
          const tool = visibleTools[idx];
          const selected = isToolSelected(selectedTools, ref);
          return (
            <label
              key={tool.name}
              className={`flex cursor-pointer items-start gap-3 px-3 py-2 rounded-lg transition-colors ${
                selected
                  ? "bg-purple-50 border border-purple-300 dark:bg-purple-950 dark:border-purple-700"
                  : "bg-card border border-border hover:bg-muted"
              }`}
            >
              <Checkbox checked={selected} onCheckedChange={() => onToggleTool(ref)} className="mt-0.5" />
              <div className="min-w-0 flex-1">
                <p
                  className={`text-sm font-medium leading-tight ${selected ? "text-purple-800 dark:text-purple-200" : "text-foreground"}`}
                >
                  {tool.name}
                </p>
                {tool.description && (
                  <p className="text-xs text-muted-foreground mt-0.5 leading-tight line-clamp-2">{tool.description}</p>
                )}
              </div>
            </label>
          );
        })}
      </div>
    );
  };

  return (
    <div className="border border-border rounded-lg overflow-hidden">
      <div className="flex items-center gap-3 pl-4 bg-muted">
        <Checkbox
          {...checkboxStateProps(groupState)}
          disabled={visibleRefs.length === 0}
          // Base UI renders a span, which the global button cursor rule does not cover.
          className="cursor-pointer data-disabled:cursor-not-allowed"
          aria-label={`${searchActive ? "Select matching tools" : "Select all tools"} on ${serverName}`}
          onCheckedChange={(checked) => onSetToolsSelected(visibleRefs, checked)}
        />
        <button
          type="button"
          aria-expanded={expanded}
          className="flex-1 flex items-center justify-between pr-4 py-3 hover:bg-accent transition-colors"
          onClick={onToggleExpanded}
        >
          <span className="text-sm font-medium text-foreground flex items-center gap-2">
            {serverName}
            {selectedCount > 0 && (
              <span className="ml-1 text-xs text-purple-600 font-semibold dark:text-purple-400">
                {selectedCount} selected
              </span>
            )}
          </span>
          <span className="text-muted-foreground text-xs">{expanded ? "▲" : "▼"}</span>
        </button>
      </div>
      {expanded && <div className="p-2">{renderBody()}</div>}
    </div>
  );
}

interface CreateToolsetModalProps {
  open: boolean;
  onClose: () => void;
  onSave: (name: string, description: string | undefined, tools: MCPToolsetTool[]) => Promise<void>;
  initialToolset?: MCPToolset;
}

const toolsetFormDefaults = (toolset?: MCPToolset): ToolsetFormValues => ({
  toolset_name: toolset?.toolset_name || "",
  description: toolset?.description || "",
});

function CreateToolsetModal({ open, onClose, onSave, initialToolset }: CreateToolsetModalProps) {
  const form = useZodForm(toolsetSchema, { defaultValues: toolsetFormDefaults(initialToolset) });
  const [selectedTools, setSelectedTools] = useState<MCPToolsetTool[]>(initialToolset?.tools || []);
  const [saving, setSaving] = useState(false);
  const [search, setSearch] = useState("");
  const [debouncedSearch] = useDebouncedValue(search, { wait: SEARCH_DEBOUNCE_WAIT_MS });
  const [expandedServerIds, setExpandedServerIds] = useState<ReadonlySet<string>>(() => new Set());
  const { data: mcpServers = [], isSuccess: serversLoaded } = useMCPServers();
  const serverIds = React.useMemo(() => mcpServers.map((s) => s.server_id), [mcpServers]);
  // Searching needs every server's catalog; otherwise only expanded servers are fetched.
  const fetchServerIds = React.useMemo(
    () => (debouncedSearch.trim() ? new Set(serverIds) : expandedServerIds),
    [debouncedSearch, serverIds, expandedServerIds],
  );
  const toolsByServer = useMCPServerTools(serverIds, fetchServerIds);
  const serverPrefixById = React.useMemo(
    () => new Map(mcpServers.map((s) => [s.server_id, s.alias || s.server_name || s.server_id])),
    [mcpServers],
  );

  React.useEffect(() => {
    if (open) {
      form.reset(toolsetFormDefaults(initialToolset));
      setSelectedTools(initialToolset?.tools || []);
      setSearch("");
      setExpandedServerIds(new Set());
    }
  }, [open, initialToolset, form]);

  const handleToggleTool = (tool: MCPToolsetTool) => {
    setSelectedTools((prev) => toggleTool(prev, tool));
  };

  const handleSetToolsSelected = (tools: MCPToolsetTool[], selected: boolean) => {
    setSelectedTools((prev) => (selected ? selectTools(prev, tools) : deselectTools(prev, tools)));
  };

  const handleToggleExpanded = (serverId: string) => {
    setExpandedServerIds((prev) => {
      const next = new Set(prev);
      if (next.has(serverId)) {
        next.delete(serverId);
      } else {
        next.add(serverId);
      }
      return next;
    });
  };

  const handleSubmit = async (values: ToolsetFormValues) => {
    setSaving(true);
    try {
      await onSave(values.toolset_name, values.description, selectedTools);
      onClose();
    } finally {
      setSaving(false);
    }
  };

  const searchActive = search.trim() !== "";
  const serverViews = buildServerViews(mcpServers, toolsByServer, search);
  const { matchingTools, pendingServerCount } = searchActive
    ? summarizeSearch(serverViews)
    : { matchingTools: [], pendingServerCount: 0 };
  const matchingState = selectionState(selectedTools, matchingTools);

  return (
    <Dialog open={open} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[960px]">
        <DialogHeader>
          <DialogTitle>{initialToolset ? "Edit Toolset" : "New Toolset"}</DialogTitle>
        </DialogHeader>
        <form onSubmit={(event) => event.preventDefault()} className="mt-2">
          <FieldGroup className="mb-4 flex-row gap-4">
            <FormField control={form.control} name="toolset_name" label="Toolset Name" className="flex-1">
              {(field) => <Input {...field} placeholder="e.g. github-linear-tools" />}
            </FormField>
            <FormField control={form.control} name="description" label="Description" className="flex-1">
              {(field) => <Input {...field} placeholder="Optional description" />}
            </FormField>
          </FieldGroup>
        </form>

        <div className="flex gap-4 mt-2" style={{ minHeight: 360 }}>
          {/* Left panel: Available Tools */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm font-semibold text-foreground">Available Tools</p>
            </div>
            <InputGroup className="mb-2">
              <InputGroupInput
                placeholder="Search tools or servers..."
                aria-label="Search tools or servers"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
              {search && (
                <InputGroupAddon align="inline-end">
                  <InputGroupButton size="icon-xs" aria-label="Clear search" onClick={() => setSearch("")}>
                    <X />
                  </InputGroupButton>
                </InputGroupAddon>
              )}
            </InputGroup>
            {searchActive && (
              <div className="mb-2 flex items-center justify-between gap-2">
                <p className="text-xs text-muted-foreground" aria-live="polite">
                  {matchingTools.length} matching {matchingTools.length === 1 ? "tool" : "tools"}
                  {pendingServerCount > 0 &&
                    ` · loading ${pendingServerCount} ${pendingServerCount === 1 ? "server" : "servers"}…`}
                </p>
                <div className="flex gap-1 shrink-0">
                  <Button
                    size="xs"
                    variant="outline"
                    disabled={matchingTools.length === 0 || matchingState === "all"}
                    onClick={() => handleSetToolsSelected(matchingTools, true)}
                  >
                    Select all matching
                  </Button>
                  <Button
                    size="xs"
                    variant="outline"
                    disabled={matchingState === "none"}
                    onClick={() => handleSetToolsSelected(matchingTools, false)}
                  >
                    Deselect all matching
                  </Button>
                </div>
              </div>
            )}
            <div className="space-y-2 overflow-y-auto" style={{ maxHeight: 300 }}>
              {serverViews.length === 0 ? (
                <p className="text-muted-foreground text-sm">
                  {mcpServers.length === 0 ? "No MCP servers configured" : "No tools or servers match your search"}
                </p>
              ) : (
                serverViews.map(({ server, serverName, state, visibleTools }) => (
                  <ServerToolGroup
                    key={server.server_id}
                    serverId={server.server_id}
                    serverName={serverName}
                    state={state}
                    visibleTools={visibleTools}
                    expanded={searchActive || expandedServerIds.has(server.server_id)}
                    searchActive={searchActive}
                    selectedTools={selectedTools}
                    onToggleExpanded={() => handleToggleExpanded(server.server_id)}
                    onToggleTool={handleToggleTool}
                    onSetToolsSelected={handleSetToolsSelected}
                  />
                ))
              )}
            </div>
          </div>

          {/* Divider */}
          <div className="w-px bg-border shrink-0" />

          {/* Right panel: Your Toolset */}
          <div className="w-72 shrink-0">
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm font-semibold text-foreground">
                Your Toolset{" "}
                <span className="text-xs font-normal text-muted-foreground">({selectedTools.length} tools)</span>
              </p>
              <Button
                size="xs"
                variant="ghost"
                disabled={selectedTools.length === 0}
                onClick={() => setSelectedTools([])}
              >
                Clear all
              </Button>
            </div>
            <div className="space-y-1 overflow-y-auto" style={{ maxHeight: 340 }}>
              {selectedTools.length === 0 ? (
                <p className="text-muted-foreground text-sm">No tools added yet</p>
              ) : (
                selectedTools.map((tool) => {
                  const toolName = displayToolName(serverPrefixById.get(tool.server_id), tool.tool_name);
                  const issue = serversLoaded ? selectedToolIssue(tool, serverPrefixById, toolsByServer) : undefined;
                  return (
                    <button
                      key={`${tool.server_id}:${tool.tool_name}`}
                      type="button"
                      aria-label={`Remove ${toolName}`}
                      onClick={() => handleToggleTool(tool)}
                      className="w-full flex items-center justify-between px-3 py-1.5 rounded-lg border border-purple-200 bg-purple-50 hover:bg-destructive/10 hover:border-destructive/20 group transition-colors dark:border-purple-800 dark:bg-purple-950"
                    >
                      <div className="min-w-0 text-left">
                        <span className="text-xs font-medium text-purple-800 group-hover:text-destructive truncate block dark:text-purple-200">
                          {toolName}
                        </span>
                        {issue ? (
                          <span className="text-[10px] text-warning truncate block">{issue}</span>
                        ) : (
                          <span className="text-[10px] text-purple-400 truncate block dark:text-purple-500">
                            {tool.server_id.slice(0, 8)}…
                          </span>
                        )}
                      </div>
                      <span className="ml-2 text-purple-300 group-hover:text-destructive text-xs shrink-0 dark:text-purple-600">
                        ✕
                      </span>
                    </button>
                  );
                })
              )}
            </div>
          </div>
        </div>

        <div className="flex justify-end gap-2 mt-4 pt-4 border-t border-border">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={() => void form.handleSubmit(handleSubmit)()} disabled={saving} aria-busy={saving}>
            {saving && <UiLoadingSpinner className="size-4" />}
            {initialToolset ? "Save Changes" : "Create Toolset"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function ToolsetsEmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">No toolsets yet</div>
      <div className="text-sm text-muted-foreground">
        Create a toolset to give keys and teams a curated set of MCP tools.
      </div>
    </div>
  );
}

function ToolsetUsageGuide() {
  const [copied, setCopied] = useState(false);
  const gatewayBaseUrl = getGatewayBaseUrl();

  const snippet = `{
  "mcpServers": {
    "my-toolset": {
      "url": "${gatewayBaseUrl}/toolset/<toolset-name>/mcp",
      "headers": { "x-litellm-api-key": "Bearer <your-api-key>" }
    }
  }
}`;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(snippet);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // ignore
    }
  };

  return (
    <div className="mb-6 rounded-lg border border-border bg-muted px-5 py-4">
      <p className="text-sm font-medium text-foreground mb-1">How toolsets work</p>
      <p className="text-sm text-muted-foreground mb-3">
        Create a toolset, assign it to a key via{" "}
        <span className="font-medium text-foreground">API Keys → Edit Key → MCP Servers</span>, then point your MCP
        client at the toolset URL. The client only sees the tools you picked.
      </p>
      <div className="text-xs text-muted-foreground mb-1">Claude Code / Cursor config</div>
      <div className="relative">
        <pre className="bg-card border border-border rounded-sm px-4 py-3 text-xs font-mono text-foreground overflow-x-auto leading-relaxed pr-14">
          {snippet}
        </pre>
        <button
          type="button"
          onClick={copy}
          className="absolute top-2 right-2 px-2 py-1 text-xs rounded-sm border bg-card hover:bg-muted text-muted-foreground hover:text-foreground border-border transition-colors"
        >
          {copied ? "✓" : "copy"}
        </button>
      </div>
    </div>
  );
}

export function MCPToolsetsTab({ accessToken, userRole }: MCPToolsetsTabProps) {
  const queryClient = useQueryClient();
  const { data: toolsets = [], isLoading } = useMCPToolsets();
  const { data: mcpServers = [] } = useMCPServers();
  const [createOpen, setCreateOpen] = useState(false);
  const [editToolset, setEditToolset] = useState<MCPToolset | null>(null);
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  const isAdmin = userRole === "Admin" || userRole === "proxy_admin";

  const handleCreate = async (name: string, description: string | undefined, tools: MCPToolsetTool[]) => {
    if (!accessToken) return;
    await createMCPToolset(accessToken, { toolset_name: name, description, tools });
    toast.success("Toolset created");
    queryClient.invalidateQueries({ queryKey: ["mcpToolsets"] });
  };

  const handleUpdate = async (name: string, description: string | undefined, tools: MCPToolsetTool[]) => {
    if (!accessToken || !editToolset) return;
    await updateMCPToolset(accessToken, { toolset_id: editToolset.toolset_id, toolset_name: name, description, tools });
    toast.success("Toolset updated");
    queryClient.invalidateQueries({ queryKey: ["mcpToolsets"] });
    setEditToolset(null);
  };

  const handleDelete = async () => {
    if (!accessToken || !deleteId) return;
    setDeleting(true);
    try {
      await deleteMCPToolset(accessToken, deleteId);
      toast.success("Toolset deleted");
      queryClient.invalidateQueries({ queryKey: ["mcpToolsets"] });
      setDeleteId(null);
    } finally {
      setDeleting(false);
    }
  };

  const serverPrefixById = React.useMemo(
    () => new Map(mcpServers.map((s) => [s.server_id, s.alias || s.server_name || s.server_id])),
    [mcpServers],
  );
  const [sorting, setSorting] = useState<SortingState>([]);
  const columns = React.useMemo(() => {
    const deps = {
      isAdmin,
      serverPrefixById,
      onEditClick: setEditToolset,
      onDeleteClick: setDeleteId,
    };
    return getMCPToolsetTableColumns(deps);
  }, [isAdmin, serverPrefixById]);

  return (
    <div className="mt-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-medium text-foreground">MCP Toolsets</h3>
          <p className="text-muted-foreground text-sm">
            Curated collections of tools from one or more MCP servers. Assign toolsets to keys and teams via the MCP
            permissions dropdown.
          </p>
        </div>
        {isAdmin && (
          <Button onClick={() => setCreateOpen(true)}>
            <Plus />
            New Toolset
          </Button>
        )}
      </div>

      <ToolsetUsageGuide />

      <DataTable
        data={toolsets}
        paginationMode="client"
        columns={columns}
        getRowId={(toolset, index) => toolset.toolset_id || String(index)}
        sortingMode="client"
        sorting={sorting}
        onSortingChange={setSorting}
        isLoading={isLoading}
        loadingMessage="Loading toolsets…"
        noDataMessage={<ToolsetsEmptyState />}
        size="compact"
      />

      <CreateToolsetModal open={createOpen} onClose={() => setCreateOpen(false)} onSave={handleCreate} />

      {editToolset && (
        <CreateToolsetModal
          open={!!editToolset}
          onClose={() => setEditToolset(null)}
          onSave={handleUpdate}
          initialToolset={editToolset}
        />
      )}

      <Dialog open={!!deleteId} onOpenChange={(open) => !open && setDeleteId(null)}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Delete Toolset</DialogTitle>
          </DialogHeader>
          <p>
            Are you sure you want to delete this toolset? Keys and teams using it will lose access to the scoped tools.
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteId(null)}>
              Cancel
            </Button>
            <Button onClick={handleDelete} variant="destructive" disabled={deleting} aria-busy={deleting}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
