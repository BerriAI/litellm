import React, { useState, useCallback } from "react";
import { z } from "zod/v4";
import { SortingState } from "@tanstack/react-table";
import { Inbox, Plus, X } from "lucide-react";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useQueryClient } from "@tanstack/react-query";
import { DataTable } from "@/components/shared/DataTable";
import {
  createMCPToolset,
  updateMCPToolset,
  deleteMCPToolset,
  listMCPTools,
  getProxyBaseUrl,
} from "@/components/networking";
import { MCPToolset, MCPToolsetTool } from "@/components/mcp_tools/types";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";
import { displayToolName, getMCPToolsetTableColumns } from "./MCPToolsetTableColumns";
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

interface MCPToolListProps {
  serverId: string;
  serverName: string;
  accessToken: string | null;
  selectedTools: MCPToolsetTool[];
  onToggle: (tool: MCPToolsetTool) => void;
}

interface ToolEntry {
  name: string;
  description?: string;
}

function MCPToolList({ serverId, serverName, accessToken, selectedTools, onToggle }: MCPToolListProps) {
  const [tools, setTools] = useState<ToolEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);

  const selectedSet = new Set(selectedTools.filter((t) => t.server_id === serverId).map((t) => t.tool_name));

  const fetchTools = useCallback(async () => {
    if (!accessToken || tools.length > 0) return;
    setLoading(true);
    try {
      const result = await listMCPTools(accessToken, serverId);
      const toolList = Array.isArray(result) ? result : result?.tools ?? [];
      setTools(toolList.map((t: any) => ({ name: t.name ?? t.tool_name ?? t, description: t.description ?? "" })));
    } catch {
      setTools([]);
    } finally {
      setLoading(false);
    }
  }, [accessToken, serverId, tools.length]);

  const handleToggle = () => {
    if (!expanded) fetchTools();
    setExpanded(!expanded);
  };

  return (
    <div className="border border-border rounded-lg overflow-hidden">
      <button
        type="button"
        className="w-full flex items-center justify-between px-4 py-3 bg-muted hover:bg-accent transition-colors"
        onClick={handleToggle}
      >
        <span className="text-sm font-medium text-foreground flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-info shrink-0" />
          {serverName}
          {selectedSet.size > 0 && (
            <span className="ml-1 text-xs text-purple-600 font-semibold dark:text-purple-400">
              {selectedSet.size} selected
            </span>
          )}
        </span>
        <span className="text-muted-foreground text-xs">{expanded ? "▲" : "▼"}</span>
      </button>
      {expanded && (
        <div className="p-2">
          {loading ? (
            <div className="flex justify-center py-3">
              <UiLoadingSpinner className="size-4" />
            </div>
          ) : tools.length === 0 ? (
            <p className="text-xs text-muted-foreground px-2 py-2">No tools found for this server.</p>
          ) : (
            <div className="flex flex-col gap-1">
              {tools.map((tool) => {
                const selected = selectedSet.has(tool.name);
                return (
                  <button
                    key={tool.name}
                    type="button"
                    onClick={() => onToggle({ server_id: serverId, tool_name: tool.name })}
                    className={`flex items-start justify-between px-3 py-2 rounded-lg text-left transition-colors ${
                      selected
                        ? "bg-purple-50 border border-purple-300 dark:bg-purple-950 dark:border-purple-700"
                        : "bg-card border border-border hover:bg-muted"
                    }`}
                  >
                    <div className="min-w-0 flex-1">
                      <p
                        className={`text-sm font-medium leading-tight ${selected ? "text-purple-800 dark:text-purple-200" : "text-foreground"}`}
                      >
                        {tool.name}
                      </p>
                      {tool.description && (
                        <p className="text-xs text-muted-foreground mt-0.5 leading-tight line-clamp-2">
                          {tool.description}
                        </p>
                      )}
                    </div>
                    {selected && (
                      <span className="text-purple-500 text-xs font-semibold ml-2 shrink-0 mt-0.5 dark:text-purple-400">
                        ✓
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface CreateToolsetModalProps {
  open: boolean;
  onClose: () => void;
  onSave: (name: string, description: string | undefined, tools: MCPToolsetTool[]) => Promise<void>;
  accessToken: string | null;
  initialToolset?: MCPToolset;
}

function CreateToolsetModal({ open, onClose, onSave, accessToken, initialToolset }: CreateToolsetModalProps) {
  const form = useZodForm(toolsetSchema, {
    defaultValues: {
      toolset_name: initialToolset?.toolset_name || "",
      description: initialToolset?.description || "",
    },
  });
  const [selectedTools, setSelectedTools] = useState<MCPToolsetTool[]>(initialToolset?.tools || []);
  const [saving, setSaving] = useState(false);
  const [serverSearch, setServerSearch] = useState("");
  const { data: mcpServers = [] } = useMCPServers();
  const serverPrefixById = React.useMemo(
    () => new Map(mcpServers.map((s) => [s.server_id, s.alias || s.server_name || s.server_id])),
    [mcpServers],
  );

  React.useEffect(() => {
    if (open) {
      form.reset({
        toolset_name: initialToolset?.toolset_name || "",
        description: initialToolset?.description || "",
      });
      setSelectedTools(initialToolset?.tools || []);
      setServerSearch("");
    }
  }, [open, initialToolset, form]);

  const handleToggleTool = (tool: MCPToolsetTool) => {
    setSelectedTools((prev) => {
      const exists = prev.some((t) => t.server_id === tool.server_id && t.tool_name === tool.tool_name);
      return exists
        ? prev.filter((t) => !(t.server_id === tool.server_id && t.tool_name === tool.tool_name))
        : [...prev, tool];
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

  const filteredServers = mcpServers.filter((s) => {
    const q = serverSearch.toLowerCase();
    return !q || (s.alias || "").toLowerCase().includes(q) || (s.server_name || "").toLowerCase().includes(q);
  });

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
                placeholder="Search MCP servers..."
                value={serverSearch}
                onChange={(e) => setServerSearch(e.target.value)}
              />
              {serverSearch && (
                <InputGroupAddon align="inline-end">
                  <InputGroupButton size="icon-xs" aria-label="Clear search" onClick={() => setServerSearch("")}>
                    <X />
                  </InputGroupButton>
                </InputGroupAddon>
              )}
            </InputGroup>
            <div className="space-y-2 overflow-y-auto" style={{ maxHeight: 300 }}>
              {filteredServers.length === 0 ? (
                <p className="text-muted-foreground text-sm">
                  {mcpServers.length === 0 ? "No MCP servers configured" : "No servers match your search"}
                </p>
              ) : (
                filteredServers.map((server) => (
                  <MCPToolList
                    key={server.server_id}
                    serverId={server.server_id}
                    serverName={server.alias || server.server_name || server.server_id}
                    accessToken={accessToken}
                    selectedTools={selectedTools}
                    onToggle={handleToggleTool}
                  />
                ))
              )}
            </div>
          </div>

          {/* Divider */}
          <div className="w-px bg-border shrink-0" />

          {/* Right panel: Your Toolset */}
          <div className="w-72 shrink-0">
            <p className="text-sm font-semibold text-foreground mb-2 block">
              Your Toolset{" "}
              <span className="text-xs font-normal text-muted-foreground">({selectedTools.length} tools)</span>
            </p>
            <div className="space-y-1 overflow-y-auto" style={{ maxHeight: 340 }}>
              {selectedTools.length === 0 ? (
                <p className="text-muted-foreground text-sm">No tools added yet</p>
              ) : (
                selectedTools.map((tool, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleToggleTool(tool)}
                    className="w-full flex items-center justify-between px-3 py-1.5 rounded-lg border border-purple-200 bg-purple-50 hover:bg-destructive/10 hover:border-destructive/20 group transition-colors dark:border-purple-800 dark:bg-purple-950"
                  >
                    <div className="min-w-0 text-left">
                      <span className="text-xs font-medium text-purple-800 group-hover:text-destructive truncate block dark:text-purple-200">
                        {displayToolName(serverPrefixById.get(tool.server_id), tool.tool_name)}
                      </span>
                      <span className="text-[10px] text-purple-400 truncate block dark:text-purple-500">
                        {tool.server_id.slice(0, 8)}…
                      </span>
                    </div>
                    <span className="ml-2 text-purple-300 group-hover:text-destructive text-xs shrink-0 dark:text-purple-600">
                      ✕
                    </span>
                  </button>
                ))
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
  const proxyBaseUrl = getProxyBaseUrl();

  const snippet = `{
  "mcpServers": {
    "my-toolset": {
      "url": "${proxyBaseUrl}/toolset/<toolset-name>/mcp",
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

      <CreateToolsetModal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        onSave={handleCreate}
        accessToken={accessToken}
      />

      {editToolset && (
        <CreateToolsetModal
          open={!!editToolset}
          onClose={() => setEditToolset(null)}
          onSave={handleUpdate}
          accessToken={accessToken}
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
