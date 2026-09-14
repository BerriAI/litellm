"use client";

import type { ColumnDef } from "@tanstack/react-table";
import { ChevronDown, ChevronRight, Search } from "lucide-react";
import { useState } from "react";

import { DataTable } from "@/components/shared/DataTable";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Input } from "@/components/ui/input";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import type { components } from "@/lib/http/schema";

type Entry = components["schemas"]["MemoryEntry"];

function contributor(entry: Entry) {
  if (entry.actor_name) return entry.actor_name;
  if (!entry.actor) return "Unknown contributor";
  return /^[a-f0-9]{64}$/.test(entry.actor) ? "Unlinked key" : entry.actor;
}

function memoryDetails(entry: Entry) {
  const details = {
    "Contributed by": contributor(entry),
    Evidence: entry.evidence,
    "When to use": entry.when_to_use,
    Source: entry.source,
    Kind: entry.kind,
    Certainty: entry.certainty,
    Context: entry.scope,
    Created: entry.created_at ? new Date(entry.created_at).toLocaleString() : undefined,
    Updated: new Date(entry.updated_at).toLocaleString(),
    "Memory ID": entry.memory_id,
  };
  return Object.entries(details).filter(([, value]) => !!value);
}

export function MemoryEntriesTable({
  entries,
  query,
  onQueryChange,
  loading,
  empty,
  footer,
  readOnly,
  canEdit,
  busy,
  onEdit,
  onDelete,
}: Readonly<{
  entries: Entry[];
  query: string;
  onQueryChange: (query: string) => void;
  loading: boolean;
  empty: React.ReactNode;
  footer: React.ReactNode;
  readOnly: boolean;
  canEdit: boolean;
  busy: boolean;
  onEdit: (entry: Entry) => void;
  onDelete: (entry: Entry) => void;
}>) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = entries.find((entry) => entry.memory_id === selectedId);
  const columns: ColumnDef<Entry>[] = [
    {
      id: "memory",
      header: "Memory",
      size: 600,
      cell: ({ row: { original: entry } }) => (
        <div className="min-w-48 space-y-1 py-1">
          <Button
            variant="link"
            className="h-auto justify-start whitespace-normal p-0 text-left font-medium text-foreground"
            aria-label={`Details for ${entry.title}`}
            onClick={() => setSelectedId(entry.memory_id)}
          >
            {entry.title}
          </Button>
          <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">{entry.content}</p>
        </div>
      ),
    },
    {
      id: "contributor",
      header: "Contributed by",
      size: 180,
      cell: ({ row }) => <span className="break-words text-sm">{contributor(row.original)}</span>,
    },
    {
      id: "saved",
      header: "Updated",
      size: 130,
      cell: ({ row }) => (
        <time dateTime={row.original.updated_at} className="whitespace-nowrap text-xs text-muted-foreground">
          {new Date(row.original.updated_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}
          <span className="block">
            {new Date(row.original.updated_at).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}
          </span>
        </time>
      ),
    },
    {
      id: "details",
      header: () => <span className="sr-only">Details</span>,
      size: 36,
      cell: () => <ChevronRight className="size-4 text-muted-foreground" aria-hidden="true" />,
    },
  ];
  return (
    <>
      <DataTable
        data={entries}
        columns={columns}
        getRowId={(entry) => entry.memory_id}
        onRowClick={(entry) => setSelectedId(entry.memory_id)}
        isLoading={loading}
        loadingMessage="Loading memories"
        noDataMessage={empty}
        sortingMode="none"
        paginationMode="none"
        filterMode="none"
        toolbar={() => (
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="relative w-full sm:w-64">
              <Search className="pointer-events-none absolute left-2.5 top-2 size-4 text-muted-foreground" />
              <Input
                aria-label="Search memories"
                placeholder="Search memories"
                className="h-8 pl-8"
                value={query}
                onChange={(event) => onQueryChange(event.target.value)}
              />
            </div>
            <span className="text-xs text-muted-foreground">Newest first</span>
          </div>
        )}
        paginationSlot={() => footer}
      />
      <Sheet open={!!selected} onOpenChange={(open) => !open && setSelectedId(null)}>
        <SheetContent className="overflow-y-auto sm:max-w-xl">
          <SheetHeader className="pr-12">
            <SheetTitle>{selected?.title}</SheetTitle>
          </SheetHeader>
          {selected && (
            <div className="space-y-6 px-4 pb-6">
              <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">{selected.content}</p>
              <p className="text-xs text-muted-foreground">Contributed by {contributor(selected)}</p>
              <Collapsible>
                <CollapsibleTrigger render={<Button variant="ghost" size="sm" className="gap-1" />}>
                  <ChevronDown className="size-3" /> Details
                </CollapsibleTrigger>
                <CollapsibleContent>
                  <dl className="mt-3 grid gap-x-5 gap-y-3 border-t pt-4 text-xs sm:grid-cols-[auto_1fr]">
                    {memoryDetails(selected).map(([label, value]) => (
                      <div key={label} className="contents">
                        <dt className="text-muted-foreground">{label}</dt>
                        <dd className="min-w-0 whitespace-pre-wrap break-words">{value}</dd>
                      </div>
                    ))}
                  </dl>
                </CollapsibleContent>
              </Collapsible>
              {!readOnly && (
                <div className="flex gap-2 border-t pt-4">
                  <Button variant="outline" size="sm" disabled={busy || !canEdit} onClick={() => onEdit(selected)}>
                    Edit memory
                  </Button>
                  <Button variant="ghost" size="sm" disabled={busy} onClick={() => onDelete(selected)}>
                    Delete memory
                  </Button>
                </div>
              )}
            </div>
          )}
        </SheetContent>
      </Sheet>
    </>
  );
}
