"use client";

import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Brain } from "lucide-react";
import { type FormEvent, useState } from "react";

import { useKeys } from "@/app/(dashboard)/hooks/keys/useKeys";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { fetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { toast } from "@/lib/toast";

import { MemoryPreference } from "./MemorySettings";
import { MemoryKeyPicker } from "./MemoryTargetPicker";
import { MemoryEntriesTable } from "./MemoryEntriesTable";

type Entry = components["schemas"]["MemoryEntry"];
type Capture = components["schemas"]["MemoryCapture"];
type Cursor = { before_updated_at: string; before_memory_id: string } | null;
type Status = components["schemas"]["MemoryStatus"];

function memoryDescription(status?: Status) {
  if (!status) return "What your assistants remember across conversations.";
  if (status.activation === "disabled" || !status.scope)
    return "Memory is off. Your administrator can make it available.";
  if (status.activation === "automatic") {
    if (status.active) return "Memory is on. Your administrator manages this setting.";
    return "Memory is off. Your administrator can make it available.";
  }
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

type DashboardProps = Readonly<{ userId: string; readOnly: boolean; proxyAdmin: boolean }>;

export function AutomaticMemoryEntries({ userId, readOnly, proxyAdmin }: DashboardProps) {
  const [selection, setSelection] = useState<string>();
  const keyOptions = {
    userID: proxyAdmin ? undefined : userId,
    sortBy: "created_at",
    sortOrder: "desc",
    includeTeamKeys: proxyAdmin,
    includeCreatedByKeys: proxyAdmin,
  };
  const keys = useKeys(1, 1, keyOptions);
  const keyId = selection ?? keys.data?.keys[0]?.token ?? "";
  return (
    <MemoryDashboard key={`${userId}:${keyId}`} userId={userId} keyId={keyId} readOnly={readOnly}>
      <div className="w-full space-y-1.5 sm:w-80">
        <Label htmlFor="memory-entry-key" className="text-xs text-muted-foreground">
          Key context
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
  const statusOptions = {
    queryKey: ["memoryStatus", userId, keyId],
    enabled: !!keyId,
    refetchOnMount: true,
    queryFn: async ({ signal }: { signal: AbortSignal }) =>
      (await fetchClient.GET("/v2/memory/status", { params: { query: { key_id: keyId } }, signal })).data,
  };
  const status = useQuery(statusOptions);
  const entriesOptions = {
    queryKey: ["memoryEntries", userId, keyId, query],
    enabled: !!keyId && !!status.data?.scope,
    refetchOnMount: true,
    initialPageParam: null as Cursor,
    queryFn: async ({ signal, pageParam }: { signal: AbortSignal; pageParam: Cursor }) =>
      (
        await fetchClient.GET("/v2/memory/entries", {
          params: { query: { key_id: keyId, query, limit: 20, ...pageParam } },
          signal,
        })
      ).data ?? [],
    getNextPageParam: (lastPage: Entry[]) => {
      const last = lastPage.at(-1);
      return lastPage.length === 20 && last
        ? { before_updated_at: last.updated_at, before_memory_id: last.memory_id }
        : undefined;
    },
  };
  const entries = useInfiniteQuery(entriesOptions);
  const save = useMutation({
    mutationFn: async (body: Capture) =>
      fetchClient.POST("/v2/memory/entries", { params: { query: { key_id: keyId } }, body }),
    onSuccess: () => {
      setEditing(null);
      toast.success("Memory updated");
      return cache.invalidateQueries({ queryKey: ["memoryEntries", userId] });
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
      return cache.invalidateQueries({ queryKey: ["memoryEntries", userId] });
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
        {status.data?.user_id && (
          <p className="text-sm text-muted-foreground">
            Memory for{" "}
            <span className="font-medium text-foreground">{status.data.user_name ?? status.data.user_id}</span>
            {status.data.scope === "user" && ". Shared across this user's keys in this organization."}
          </p>
        )}
      </div>
      {status.error && (
        <p role="alert" className="text-sm text-destructive">
          Could not load memory status: {status.error.message}
        </p>
      )}
      {status.data?.scope && (
        <div className="space-y-3">
          <h2 className="sr-only">Saved memories</h2>
          <MemoryEntriesTable
            entries={memories}
            query={query}
            onQueryChange={setQuery}
            loading={entries.isPending}
            readOnly={readOnly}
            canEdit={status.data.active}
            busy={busy}
            onEdit={setEditing}
            onDelete={setDeleting}
            empty={
              entries.error ? (
                <p className="text-destructive">Could not load memories: {entries.error.message}</p>
              ) : (
                <div className="flex flex-col items-center gap-2 py-8 text-center">
                  <Brain className="mb-1 size-6 text-muted-foreground" />
                  <h3 className="font-medium">{query ? "No matching memories" : "No memories yet"}</h3>
                  <p className="text-sm text-muted-foreground">{emptyDescription(query, status.data.active)}</p>
                </div>
              )
            }
            footer={
              <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
                <span className="text-xs text-muted-foreground">
                  {memories.length} memories
                  {!status.data.active && memories.length > 0 && " · Saved memories stay here while memory is off"}
                </span>
                {(entries.hasNextPage || entries.isError) && (
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={entries.isFetching}
                    onClick={() =>
                      entries.isError && !entries.isFetchNextPageError ? entries.refetch() : entries.fetchNextPage()
                    }
                  >
                    {loadMoreLabel(entries.isFetching, entries.isError)}
                  </Button>
                )}
              </div>
            }
          />
          {entries.isFetchNextPageError && (
            <p role="alert" className="text-sm text-destructive">
              Could not load more memories: {entries.error.message}
            </p>
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
