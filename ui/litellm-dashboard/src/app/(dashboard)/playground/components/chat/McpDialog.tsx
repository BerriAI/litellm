"use client";

import { useId, useState, type ReactNode } from "react";
import { Blocks, KeyRound, Search } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Field, FieldLabel, FieldLegend, FieldSet } from "@/components/ui/field";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Spinner } from "@/components/ui/spinner";
import type { MCPServer, MCPToolset } from "@/components/mcp_tools/types";
import { uiHref } from "@/utils/uiHref";
import { ByokConnect } from "./ByokConnect";
import {
  ALL_MCP_SERVERS,
  TOOLSET_PREFIX,
  allowedTools,
  toggleMcpItem,
  toggleToolLimit,
  withoutStaleLimits,
  type ToolLimits,
} from "./playgroundSettings";
import { useServerTools } from "./usePlaygroundCatalog";

type Row = Readonly<{ value: string; label: string }>;

export const serverName = (server: MCPServer): string => server.alias || server.server_name || server.server_id;

const matches = (value: string, query: string): boolean => value.toLowerCase().includes(query.trim().toLowerCase());

const needsKey = (server: MCPServer | undefined): boolean => Boolean(server?.is_byok && !server.has_user_credential);

function SearchField({
  value,
  onChange,
  placeholder,
  autoFocus,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  autoFocus: boolean;
}) {
  return (
    <InputGroup className="h-8">
      <InputGroupAddon>
        <Search />
      </InputGroupAddon>
      <InputGroupInput
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        aria-label={placeholder}
        autoFocus={autoFocus}
      />
    </InputGroup>
  );
}

function MasterRow({
  row,
  checked,
  onCheckedChange,
  active,
  onSelect,
  summary,
}: {
  row: Row;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  active: boolean;
  onSelect: () => void;
  summary?: string;
}) {
  return (
    <div data-active={active || undefined} className="flex items-center gap-1 rounded-sm pl-2 data-active:bg-accent">
      <Checkbox checked={checked} onCheckedChange={onCheckedChange} aria-label={`Use ${row.label}`} />
      <Button
        variant="ghost"
        size="sm"
        aria-pressed={active}
        onClick={onSelect}
        className="min-w-0 flex-1 justify-between px-2 font-normal hover:bg-transparent aria-pressed:bg-transparent"
      >
        <span className="truncate">{row.label}</span>
        {summary && <span className="text-xs whitespace-nowrap text-muted-foreground">{summary}</span>}
      </Button>
    </div>
  );
}

function ToolRow({
  label,
  checked,
  disabled,
  onCheckedChange,
}: {
  label: string;
  checked: boolean;
  disabled: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  const id = useId();
  return (
    <Field orientation="horizontal" className="min-h-8 gap-2 rounded-sm px-2 py-1 hover:bg-accent">
      <Checkbox id={id} checked={checked} disabled={disabled} onCheckedChange={onCheckedChange} />
      <FieldLabel htmlFor={id} className="min-w-0 flex-1 truncate font-mono text-xs font-normal">
        {label}
      </FieldLabel>
    </Field>
  );
}

function Note({ title, children, action }: { title: string; children: ReactNode; action?: ReactNode }) {
  return (
    <Empty>
      <EmptyHeader>
        <EmptyTitle className="text-sm">{title}</EmptyTitle>
        <EmptyDescription>{children}</EmptyDescription>
      </EmptyHeader>
      {action && <EmptyContent>{action}</EmptyContent>}
    </Empty>
  );
}

function RowGroup({ title, rows, render }: { title: string; rows: readonly Row[]; render: (row: Row) => ReactNode }) {
  if (rows.length === 0) return null;
  return (
    <FieldSet className="gap-0">
      <FieldLegend variant="label" className="mb-0 px-2 pt-2 pb-1 text-xs font-normal text-muted-foreground">
        {title}
      </FieldLegend>
      {rows.map(render)}
    </FieldSet>
  );
}

export function McpDialog({
  requestKey,
  servers,
  toolsets,
  isLoading,
  selection,
  toolLimits,
  onChange,
  onCredentialSaved,
  defaultOpen = false,
}: {
  requestKey: string | null;
  servers: readonly MCPServer[];
  toolsets: readonly MCPToolset[];
  isLoading: boolean;
  selection: readonly string[];
  toolLimits: ToolLimits;
  onChange: (selection: readonly string[], limits: ToolLimits) => void;
  onCredentialSaved: () => void;
  defaultOpen?: boolean;
}) {
  const [focused, setFocused] = useState<string>(selection.find((item) => item !== ALL_MCP_SERVERS) ?? ALL_MCP_SERVERS);
  const [query, setQuery] = useState("");
  const [toolQuery, setToolQuery] = useState("");
  const [replacingKey, setReplacingKey] = useState<string | null>(null);
  const serverById = new Map(servers.map((server) => [server.server_id, server]));
  const selectedServers = selection.filter((item) => serverById.has(item));
  const toolsByServer = useServerTools(
    requestKey,
    [...new Set([...selectedServers, ...(serverById.has(focused) ? [focused] : [])])].filter(
      (id) => !needsKey(serverById.get(id)),
    ),
  );
  const toolNames = (serverId: string): readonly string[] =>
    (toolsByServer[serverId]?.tools ?? []).map((tool) => tool.name);

  const serverRows: Row[] = servers.map((server) => ({ value: server.server_id, label: serverName(server) }));
  const toolsetRows: Row[] = toolsets.map((toolset) => ({
    value: `${TOOLSET_PREFIX}${toolset.toolset_id}`,
    label: toolset.toolset_name,
  }));
  const visible = (rows: readonly Row[]) => rows.filter((row) => matches(row.label, query));

  const toggle = (item: string, on: boolean) => {
    const next = toggleMcpItem(selection, item, on);
    onChange(next, withoutStaleLimits(toolLimits, next));
    if (on) setFocused(item);
  };

  const summary = (row: Row): string | undefined => {
    if (!selection.includes(row.value)) return undefined;
    const server = serverById.get(row.value);
    if (!server) return undefined;
    if (needsKey(server)) return "Needs your key";
    const tools = toolNames(row.value);
    if (tools.length === 0) return undefined;
    const allowed = allowedTools(toolLimits, row.value, tools).length;
    return allowed === tools.length ? `All ${tools.length} tools` : `${allowed} of ${tools.length} tools`;
  };

  const focus = (item: string) => {
    setFocused(item);
    setToolQuery("");
    setReplacingKey(null);
  };

  const masterRow = (row: Row) => (
    <MasterRow
      key={row.value}
      row={row}
      checked={selection.includes(row.value)}
      onCheckedChange={(on) => toggle(row.value, on)}
      active={row.value === focused}
      onSelect={() => focus(row.value)}
      summary={summary(row)}
    />
  );

  const detail = (): ReactNode => {
    if (focused === ALL_MCP_SERVERS)
      return <Note title="All MCP servers">Every server this key can reach, with all of its tools</Note>;
    const toolset = toolsets.find((item) => `${TOOLSET_PREFIX}${item.toolset_id}` === focused);
    if (toolset)
      return (
        <ScrollArea className="min-h-0 flex-1">
          <div className="flex flex-col gap-3 p-4">
            <p className="text-sm text-muted-foreground">
              {toolset.description || "A toolset's tools are fixed by whoever created it"}
            </p>
            <div className="flex flex-wrap gap-1.5">
              {toolset.tools.map((tool) => (
                <Badge key={`${tool.server_id}-${tool.tool_name}`} variant="secondary" className="font-mono">
                  {tool.tool_name}
                </Badge>
              ))}
            </div>
          </div>
        </ScrollArea>
      );
    const server = serverById.get(focused);
    if (!server) return <Note title="Nothing selected">Pick a server or toolset on the left</Note>;
    const name = serverName(server);
    if (needsKey(server) || replacingKey === focused)
      return (
        <ByokConnect
          server={server}
          name={name}
          onConnected={() => {
            setReplacingKey(null);
            onCredentialSaved();
          }}
          onCancel={needsKey(server) ? undefined : () => setReplacingKey(null)}
        />
      );
    const replaceKey = server.is_byok ? (
      <Button variant="outline" size="sm" onClick={() => setReplacingKey(focused)}>
        <KeyRound />
        Replace key
      </Button>
    ) : undefined;
    if (!selection.includes(focused))
      return (
        <Note title={name} action={replaceKey}>
          Check this server to choose which of its tools the model can call
        </Note>
      );
    const state = toolsByServer[focused];
    if (!state || state.isLoading)
      return (
        <div className="flex flex-1 items-center justify-center">
          <Spinner />
        </div>
      );
    if (state.failed)
      return (
        <Note title={name} action={replaceKey}>
          Couldn&apos;t load this server&apos;s tools
        </Note>
      );
    const tools = toolNames(focused);
    if (tools.length === 0) return <Note title={name}>This server has no tools</Note>;
    const allowed = allowedTools(toolLimits, focused, tools);
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        <div className="flex items-center gap-2 p-2">
          <SearchField
            value={toolQuery}
            onChange={setToolQuery}
            placeholder={`Search ${name} tools`}
            autoFocus={false}
          />
          <Button
            variant="ghost"
            size="sm"
            disabled={allowed.length === tools.length}
            onClick={() => onChange(selection, { ...toolLimits, [focused]: [] })}
          >
            Allow all
          </Button>
          {replaceKey}
        </div>
        <ScrollArea className="min-h-0 flex-1">
          <div className="grid grid-cols-2 p-1">
            {tools
              .filter((tool) => matches(tool, toolQuery))
              .map((tool) => (
                <ToolRow
                  key={tool}
                  label={tool}
                  checked={allowed.includes(tool)}
                  disabled={allowed.length === 1 && allowed.includes(tool)}
                  onCheckedChange={(on) =>
                    onChange(selection, toggleToolLimit(toolLimits, { server: focused, tools }, tool, on))
                  }
                />
              ))}
          </div>
        </ScrollArea>
      </div>
    );
  };

  const nothingAvailable = !isLoading && servers.length === 0 && toolsets.length === 0;

  return (
    <Dialog defaultOpen={defaultOpen} onOpenChange={(open) => !open && setReplacingKey(null)}>
      <DialogTrigger render={<InputGroupButton size="xs" aria-label="MCP" />}>
        <Blocks />
        MCP
        {selection.length > 0 && (
          <Badge variant="secondary" className="h-4 min-w-4 px-1 tabular-nums">
            {selection.length}
          </Badge>
        )}
      </DialogTrigger>
      <DialogContent className="gap-0 p-0 sm:max-w-3xl">
        <DialogHeader className="border-b border-border p-4">
          <DialogTitle>MCP servers</DialogTitle>
          <DialogDescription>
            Pick the servers the model can use, then limit their tools. A server allows every tool until you uncheck
            some.
          </DialogDescription>
        </DialogHeader>
        {nothingAvailable ? (
          <div className="h-96">
            <Note title="No MCP servers for this key">
              Add one on the <a href={uiHref("mcp-servers")}>MCP servers page</a>, or send as a key that can reach one
            </Note>
          </div>
        ) : (
          <div className="grid h-96 grid-cols-[18rem_1fr]">
            <div className="flex min-h-0 flex-col border-r border-border">
              <div className="p-2">
                <SearchField value={query} onChange={setQuery} placeholder="Search servers" autoFocus />
              </div>
              <ScrollArea className="min-h-0 flex-1">
                {isLoading ? (
                  <div className="flex justify-center p-6">
                    <Spinner />
                  </div>
                ) : (
                  <div className="p-1">
                    <RowGroup
                      title="MCP servers"
                      rows={visible([{ value: ALL_MCP_SERVERS, label: "All MCP servers" }, ...serverRows])}
                      render={masterRow}
                    />
                    <RowGroup title="Toolsets" rows={visible(toolsetRows)} render={masterRow} />
                  </div>
                )}
              </ScrollArea>
            </div>
            <div className="flex min-h-0 flex-col">{detail()}</div>
          </div>
        )}
        <DialogFooter className="m-0 border-t border-border p-3">
          <DialogClose render={<Button size="sm" />}>Done</DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
