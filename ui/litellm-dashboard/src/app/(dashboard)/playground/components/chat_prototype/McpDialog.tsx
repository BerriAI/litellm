"use client";

import { useState, type ReactNode } from "react";
import { Blocks, Search } from "lucide-react";
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
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Field, FieldLabel, FieldLegend, FieldSet } from "@/components/ui/field";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { ScrollArea } from "@/components/ui/scroll-area";
import { DEFAULT_MCP, MCP_GROUPS, SERVER_TOOLS } from "./CleanSlatePickers";

type Allowed = Readonly<Record<string, readonly string[]>>;

const INITIAL_ALLOWED: Allowed = Object.fromEntries(
  Object.entries(SERVER_TOOLS).map(([server, tools]) => [
    server,
    server === "github" ? ["get_issue", "list_pull_requests", "search_code"] : tools,
  ]),
);

function useToolSelection() {
  const [selected, setSelected] = useState<readonly string[]>(DEFAULT_MCP);
  const [allowed, setAllowed] = useState<Allowed>(INITIAL_ALLOWED);
  return {
    selected,
    allowed,
    toggle: (item: string, on: boolean) =>
      setSelected(on ? [...selected, item] : selected.filter((value) => value !== item)),
    toggleTool: (server: string, tool: string, on: boolean) =>
      setAllowed({
        ...allowed,
        [server]: on ? [...(allowed[server] ?? []), tool] : (allowed[server] ?? []).filter((t) => t !== tool),
      }),
    allowAll: (server: string) => setAllowed({ ...allowed, [server]: SERVER_TOOLS[server] ?? [] }),
  };
}

type Selection = ReturnType<typeof useToolSelection>;

function isServer(item: string): boolean {
  return item in SERVER_TOOLS;
}

function accessSummary(server: string, allowed: Allowed): string {
  const total = SERVER_TOOLS[server]?.length ?? 0;
  const count = allowed[server]?.length ?? 0;
  return count === total ? `All ${total} tools` : `${count} of ${total} tools`;
}

function matches(value: string, query: string): boolean {
  return value.toLowerCase().includes(query.trim().toLowerCase());
}

function SearchField({
  value,
  onChange,
  placeholder,
  autoFocus = true,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  autoFocus?: boolean;
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

function ToolRow({
  id,
  label,
  checked,
  onCheckedChange,
}: {
  id: string;
  label: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
}) {
  return (
    <Field orientation="horizontal" className="min-h-8 gap-2 rounded-sm px-2 py-1 hover:bg-accent">
      <Checkbox id={id} checked={checked} onCheckedChange={onCheckedChange} />
      <FieldLabel htmlFor={id} className="min-w-0 flex-1 truncate font-mono text-xs font-normal">
        {label}
      </FieldLabel>
    </Field>
  );
}

function GroupedRows({
  visible,
  renderRow,
}: {
  visible: (item: string) => boolean;
  renderRow: (item: string) => ReactNode;
}) {
  return (
    <>
      {MCP_GROUPS.map((group) => {
        const items = group.items.filter(visible);
        if (items.length === 0) return null;
        return (
          <FieldSet key={group.value} className="gap-0">
            <FieldLegend variant="label" className="mb-0 px-2 pt-2 pb-1 text-xs font-normal text-muted-foreground">
              {group.value}
            </FieldLegend>
            {items.map(renderRow)}
          </FieldSet>
        );
      })}
    </>
  );
}

function ServerTools({ selection, server, query }: { selection: Selection; server: string; query: string }) {
  return (
    <>
      {(SERVER_TOOLS[server] ?? [])
        .filter((tool) => matches(tool, query))
        .map((tool) => (
          <ToolRow
            key={tool}
            id={`tool-${server}-${tool}`}
            label={tool}
            checked={(selection.allowed[server] ?? []).includes(tool)}
            onCheckedChange={(on) => selection.toggleTool(server, tool, on)}
          />
        ))}
    </>
  );
}

function MasterRow({
  item,
  checked,
  onCheckedChange,
  active,
  onSelect,
  summary,
}: {
  item: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  active: boolean;
  onSelect: () => void;
  summary?: string;
}) {
  return (
    <div data-active={active || undefined} className="flex items-center gap-1 rounded-sm pl-2 data-active:bg-accent">
      <Checkbox checked={checked} onCheckedChange={onCheckedChange} aria-label={`Use ${item}`} />
      <Button
        variant="ghost"
        size="sm"
        aria-pressed={active}
        onClick={onSelect}
        className="min-w-0 flex-1 justify-between px-2 font-normal hover:bg-transparent aria-pressed:bg-transparent"
      >
        <span className="truncate">{item}</span>
        {summary && <span className="text-xs whitespace-nowrap text-muted-foreground">{summary}</span>}
      </Button>
    </div>
  );
}

function noToolsReason(item: string): string {
  if (item === "All MCP servers") return "Every server is allowed with all of its tools";
  if (isServer(item)) return "Check this server to choose which of its tools the model can call";
  return "A toolset's tools are fixed by whoever created it";
}

export function McpDialog({ open }: { open: boolean }) {
  const selection = useToolSelection();
  const [focused, setFocused] = useState("github");
  const [query, setQuery] = useState("");
  const [toolQuery, setToolQuery] = useState("");
  const focusedServer = isServer(focused) && selection.selected.includes(focused) ? focused : null;
  return (
    <Dialog defaultOpen={open}>
      <DialogTrigger render={<InputGroupButton size="xs" aria-label="MCP" />}>
        <Blocks />
        MCP
        {selection.selected.length > 0 && (
          <Badge variant="secondary" className="h-4 min-w-4 px-1 tabular-nums">
            {selection.selected.length}
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
        <div className="grid h-96 grid-cols-[18rem_1fr]">
          <div className="flex min-h-0 flex-col border-r border-border">
            <div className="p-2">
              <SearchField value={query} onChange={setQuery} placeholder="Search servers" />
            </div>
            <ScrollArea className="min-h-0 flex-1">
              <div className="p-1">
                <GroupedRows
                  visible={(item) => matches(item, query)}
                  renderRow={(item) => (
                    <MasterRow
                      key={item}
                      item={item}
                      checked={selection.selected.includes(item)}
                      onCheckedChange={(on) => selection.toggle(item, on)}
                      active={item === focused}
                      onSelect={() => setFocused(item)}
                      summary={
                        isServer(item) && selection.selected.includes(item)
                          ? accessSummary(item, selection.allowed)
                          : undefined
                      }
                    />
                  )}
                />
              </div>
            </ScrollArea>
          </div>
          {focusedServer ? (
            <div className="flex min-h-0 flex-col">
              <div className="flex items-center gap-2 p-2">
                <SearchField
                  value={toolQuery}
                  onChange={setToolQuery}
                  placeholder={`Search ${focusedServer} tools`}
                  autoFocus={false}
                />
                <Button variant="ghost" size="sm" onClick={() => selection.allowAll(focusedServer)}>
                  Allow all
                </Button>
              </div>
              <ScrollArea className="min-h-0 flex-1">
                <div className="grid grid-cols-2 p-1">
                  <ServerTools selection={selection} server={focusedServer} query={toolQuery} />
                </div>
              </ScrollArea>
            </div>
          ) : (
            <Empty>
              <EmptyHeader>
                <EmptyTitle className="text-sm">{focused}</EmptyTitle>
                <EmptyDescription>{noToolsReason(focused)}</EmptyDescription>
              </EmptyHeader>
            </Empty>
          )}
        </div>
        <DialogFooter className="m-0 border-t border-border p-3">
          <DialogClose render={<Button size="sm" />}>Done</DialogClose>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
