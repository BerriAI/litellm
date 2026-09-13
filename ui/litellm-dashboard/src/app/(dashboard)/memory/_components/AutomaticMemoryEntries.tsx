"use client";

import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Brain, ChevronDown, Search } from "lucide-react";
import { type FormEvent, useState } from "react";

import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { fetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { toast } from "@/lib/toast";

import { MemoryPreference } from "./MemorySettings";
import { MemoryKeyPicker } from "./MemoryTargetPicker";

type Entry = components["schemas"]["MemoryEntry"];
type Capture = components["schemas"]["MemoryCapture"];
type Status = components["schemas"]["MemoryStatus"];

function memoryDescription(status?: Status) {
  if (!status) return "What your assistants remember across conversations.";
  if (status.activation === "automatic") return "Memory is on for this key. Your administrator manages this setting.";
  if (status.activation === "disabled" || !status.scope)
    return "Memory is off. Your administrator can make it available for this key.";
  if (status.active) return "Your assistants can save and recall memories. You can turn this off at any time.";
  return "Your assistants won't save or recall memories. Turn it on whenever you're ready.";
}

function emptyDescription(query: string, active: boolean) {
  if (query) return "Try a different search.";
  if (active) return "Use your assistant as usual. What it remembers will appear here.";
  return "Turn on memory, then use your assistant as usual. Your memories will appear here.";
}

function loadMoreLabel(fetching: boolean, failed: boolean) {
  if (fetching) return "Loading…";
  return failed ? "Try again" : "Load more memories";
}

function memoryDetails(entry: Entry) {
  const details = {
    Evidence: entry.evidence,
    "When to use": entry.when_to_use,
    Source: entry.source,
    Kind: entry.kind,
    Certainty: entry.certainty,
    Context: entry.scope,
    "Saved by": entry.actor,
    Created: entry.created_at ? new Date(entry.created_at).toLocaleString() : undefined,
    Updated: new Date(entry.updated_at).toLocaleString(),
    "Memory ID": entry.memory_id,
  };
  return Object.entries(details).filter(([, value]) => !!value);
}
type DashboardProps = Readonly<{ userId: string; readOnly: boolean; proxyAdmin: boolean }>;

export function AutomaticMemoryEntries({ userId, readOnly, proxyAdmin }: DashboardProps) {
  const [selection, setSelection] = useState<string>();
  const keys = useKeys(1, 1, { userID: proxyAdmin ? undefined : userId, sortBy: "created_at", sortOrder: "desc" });
  const keyId = selection ?? keys.data?.keys[0]?.token ?? "";
  return (
    <MemoryDashboard key={`${userId}:${keyId}`} userId={userId} keyId={keyId} readOnly={readOnly}>
      <div className="w-full space-y-1.5 sm:w-80">
        <Label htmlFor="memory-entry-key" className="text-xs text-muted-foreground">
          Virtual key
        </Label>
        <MemoryKeyPicker
          inputId="memory-entry-key"
          value={keyId}
          disabled={keys.isPending}
          userId={proxyAdmin ? undefined : userId}
          onChange={setSelection}
        />
        {keys.error && (
          <p role="alert" className="text-sm text-destructive">
            {keys.error.message}
          </p>
        )}
        {keys.isSuccess && keys.data.total_count === 0 && (
          <p className="text-sm text-muted-foreground">Create a virtual key to start using memory.</p>
        )}
      </div>
    </MemoryDashboard>
  );
}

function MemoryDashboard({
  userId,
  keyId,
  readOnly,
  children,
}: Readonly<{
  userId: string;
  keyId: string;
  readOnly: boolean;
  children: React.ReactNode;
}>) {
  const cache = useQueryClient();
  const [query, setQuery] = useState("");
  const [editing, setEditing] = useState<Entry | null>(null);
  const [deleting, setDeleting] = useState<Entry | null>(null);
  const status = useQuery({
    queryKey: ["memoryStatus", userId, keyId],
    enabled: !!keyId,
    queryFn: async ({ signal }) =>
      (await fetchClient.GET("/v2/memory/status", { params: { query: { key_id: keyId } }, signal })).data,
  });
  const entriesOptions = {
    queryKey: ["memoryEntries", userId, keyId, query],
    enabled: !!keyId && !!status.data?.scope,
    initialPageParam: 0,
    queryFn: async ({ signal, pageParam }: { signal: AbortSignal; pageParam: number }) =>
      (
        await fetchClient.GET("/v2/memory/entries", {
          params: { query: { key_id: keyId, query, offset: pageParam, limit: 20 } },
          signal,
        })
      ).data ?? [],
    getNextPageParam: (lastPage: Entry[], _pages: Entry[][], offset: number) =>
      lastPage.length === 20 ? offset + 20 : undefined,
  };
  const entries = useInfiniteQuery(entriesOptions);
  const save = useMutation({
    mutationFn: async (body: Capture) =>
      fetchClient.POST("/v2/memory/entries", { params: { query: { key_id: keyId } }, body }),
    onSuccess: () => {
      setEditing(null);
      toast.success("Memory updated");
      return cache.invalidateQueries({ queryKey: ["memoryEntries", userId, keyId] });
    },
    onError: (error: Error) => toast.error(error.message),
  });
  const remove = useMutation({
    mutationFn: async (memory_id: string) =>
      fetchClient.DELETE("/v2/memory/entries/{memory_id}", {
        params: { path: { memory_id }, query: { key_id: keyId } },
      }),
    onSuccess: () => {
      setDeleting(null);
      toast.success("Memory deleted");
      return cache.invalidateQueries({ queryKey: ["memoryEntries", userId, keyId] });
    },
    onError: (error: Error) => toast.error(error.message),
  });
  const busy = save.isPending || remove.isPending;
  const memories = Array.from(
    new Map((entries.data?.pages ?? []).flat().map((entry) => [entry.memory_id, entry])).values(),
  );
  const saveCorrection = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!editing) return;
    const body: Capture = {
      key: editing.key,
      title: editing.title,
      content: editing.content,
      evidence: editing.evidence,
      when_to_use: editing.when_to_use,
      scope: editing.scope,
      kind: editing.kind,
      certainty: editing.certainty,
      source: editing.source,
      expected_revision: editing.updated_at,
    };
    save.mutate(body);
  };
  const description = memoryDescription(status.data);
  return (
    <section className="space-y-8" aria-labelledby="memory-title">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-2">
          <h1 id="memory-title" className="text-[28px] font-semibold tracking-tight">
            Memory
          </h1>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
        {status.data && !status.error && (
          <MemoryPreference userId={userId} keyId={keyId} status={status.data} readOnly={readOnly} />
        )}
        {keyId && status.isPending && <Skeleton className="h-12 w-40" aria-label="Loading memory status" />}
      </div>
      <div className="flex flex-wrap items-end justify-between gap-4">
        {children}
        {status.data?.scope && (
          <div className="relative w-full sm:w-80">
            <Search className="pointer-events-none absolute left-3 top-2.5 size-4 text-muted-foreground" />
            <Input
              aria-label="Search memories"
              placeholder="Search memories"
              className="pl-9"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>
        )}
      </div>
      {status.error && (
        <p role="alert" className="text-sm text-destructive">
          Could not load memory status: {status.error.message}
        </p>
      )}
      {status.data?.scope && (
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-4">
            <h2 className="text-base font-semibold">Saved memories</h2>
            <span className="text-xs text-muted-foreground">
              {status.data.scope !== "key" &&
                status.data.scope !== "user" &&
                `Shared with your ${status.data.scope} · `}
              Newest first
            </span>
          </div>
          {!status.data.active && memories.length > 0 && (
            <p className="text-sm text-muted-foreground">Saved memories stay here while memory is off.</p>
          )}
          {entries.isPending && (
            <div role="status" aria-label="Loading memories" className="space-y-3">
              {[0, 1, 2].map((row) => (
                <Skeleton key={row} className="h-28 w-full rounded-lg" />
              ))}
            </div>
          )}
          {entries.isSuccess && memories.length === 0 && (
            <div className="flex flex-col items-center gap-3 rounded-lg border bg-card px-6 py-12 text-center">
              <Brain className="size-7 text-muted-foreground" />
              <h3 className="font-medium">{query ? "No matching memories" : "No memories yet"}</h3>
              <p className="text-sm text-muted-foreground">{emptyDescription(query, status.data.active)}</p>
            </div>
          )}
          <ul className="space-y-3" aria-label="Saved memories">
            {memories.map((entry) => (
              <li key={entry.memory_id} className="space-y-3 rounded-lg border bg-card p-5">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <h3 className="font-medium">{entry.title}</h3>
                  <time
                    dateTime={entry.updated_at}
                    title={new Date(entry.updated_at).toLocaleString()}
                    className="text-xs text-muted-foreground"
                  >
                    {new Date(entry.updated_at).toLocaleDateString(undefined, {
                      month: "short",
                      day: "numeric",
                      year: "numeric",
                    })}
                  </time>
                </div>
                <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">{entry.content}</p>
                <Collapsible>
                  <CollapsibleTrigger
                    render={<Button variant="ghost" size="xs" className="gap-1 text-muted-foreground" />}
                    aria-label={`Details for ${entry.title}`}
                  >
                    <ChevronDown className="size-3" /> Details
                    <span className="sr-only"> for {entry.title}</span>
                  </CollapsibleTrigger>
                  <CollapsibleContent>
                    <dl className="mt-3 grid gap-x-6 gap-y-2 border-t pt-3 text-xs sm:grid-cols-[auto_1fr]">
                      {memoryDetails(entry).map(([label, value]) => (
                        <div key={label} className="contents">
                          <dt className="text-muted-foreground">{label}</dt>
                          <dd className="whitespace-pre-wrap break-all">{value}</dd>
                        </div>
                      ))}
                    </dl>
                    {!readOnly && (
                      <div className="mt-3 flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={busy || !status.data?.active}
                          onClick={() => setEditing(entry)}
                        >
                          Edit memory
                        </Button>
                        <Button variant="ghost" size="sm" disabled={busy} onClick={() => setDeleting(entry)}>
                          Delete memory
                        </Button>
                      </div>
                    )}
                  </CollapsibleContent>
                </Collapsible>
              </li>
            ))}
          </ul>
          {entries.error && (
            <p role="alert" className="text-sm text-destructive">
              Could not load memories: {entries.error.message}
            </p>
          )}
          {(entries.hasNextPage || entries.isError) && (
            <div className="flex justify-center">
              <Button
                variant="outline"
                disabled={entries.isFetching}
                onClick={() =>
                  entries.isError && !entries.isFetchNextPageError ? entries.refetch() : entries.fetchNextPage()
                }
              >
                {loadMoreLabel(entries.isFetching, entries.isError)}
              </Button>
            </div>
          )}
        </div>
      )}
      <Dialog
        open={!!editing}
        onOpenChange={(open) => {
          if (!open && !busy) setEditing(null);
        }}
      >
        <DialogContent className="sm:max-w-xl">
          <DialogHeader>
            <DialogTitle>Edit memory</DialogTitle>
          </DialogHeader>
          {editing && (
            <form className="space-y-4" onSubmit={saveCorrection}>
              <div className="space-y-2">
                <Label htmlFor="memory-edit-content">Correct this memory</Label>
                <Textarea
                  id="memory-edit-content"
                  value={editing.content}
                  required
                  maxLength={8000}
                  disabled={busy}
                  rows={6}
                  onChange={(event) => setEditing({ ...editing, content: event.target.value })}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="memory-edit-evidence">Evidence</Label>
                <Textarea
                  id="memory-edit-evidence"
                  value={editing.evidence}
                  required
                  maxLength={2000}
                  disabled={busy}
                  onChange={(event) => setEditing({ ...editing, evidence: event.target.value })}
                />
              </div>
              <div className="flex justify-end gap-2">
                <Button type="button" variant="outline" disabled={busy} onClick={() => setEditing(null)}>
                  Cancel
                </Button>
                <Button type="submit" disabled={busy || !status.data?.active}>
                  Save correction
                </Button>
              </div>
            </form>
          )}
        </DialogContent>
      </Dialog>
      <DeleteResourceModal
        isOpen={!!deleting}
        onCancel={() => setDeleting(null)}
        onOk={() => {
          if (deleting) remove.mutate(deleting.memory_id);
        }}
        title="Delete memory"
        message={`Delete ${deleting?.title ?? "this memory"}?`}
        confirmLoading={remove.isPending}
      />
    </section>
  );
}
