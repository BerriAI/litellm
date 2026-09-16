"use client";

import { useDebouncedValue } from "@tanstack/react-pacer/debouncer";
import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Brain } from "lucide-react";
import { type FormEvent, useState } from "react";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { fetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { toast } from "@/lib/toast";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { MemoryTeamPicker, MemoryUserPicker } from "./MemoryTargetPicker";
import { MemoryEntriesTable } from "./MemoryEntriesTable";

type Entry = components["schemas"]["MemoryEntry"];
type Capture = components["schemas"]["MemoryCapture"];
type Cursor = { before_updated_at: string; before_memory_id: string } | null;

export function AutomaticMemoryEntries({
  userId,
  readOnly,
  proxyAdmin,
}: Readonly<{ userId: string; readOnly: boolean; proxyAdmin: boolean }>) {
  const cache = useQueryClient();
  const [search, setSearch] = useState("");
  const [query] = useDebouncedValue(search, { wait: DEBOUNCE_WAIT_MS });
  const [teamId, setTeamId] = useState("");
  const [filterUserId, setFilterUserId] = useState("");
  const [editing, setEditing] = useState<Entry | null>(null);
  const [deleting, setDeleting] = useState<Entry | null>(null);
  const status = useQuery({
    queryKey: ["memoryStatus", userId, readOnly, proxyAdmin],
    queryFn: async ({ signal }) => (await fetchClient.GET("/memory/v2/status", { signal })).data,
  });
  const entries = useInfiniteQuery({
    queryKey: ["memoryEntries", userId, query, teamId, filterUserId, status.data?.team_ids, status.data?.admin_view],
    enabled: status.isSuccess,
    initialPageParam: null as Cursor,
    queryFn: async ({ signal, pageParam }) =>
      (
        await fetchClient.GET("/memory/v2/entries", {
          params: {
            query: {
              query,
              limit: 20,
              team_id: teamId || undefined,
              user_id: filterUserId || undefined,
              ...pageParam,
            },
          },
          signal,
        })
      ).data ?? [],
    getNextPageParam: (lastPage: Entry[]) => {
      const last = lastPage.at(-1);
      return lastPage.length === 20 && last
        ? { before_updated_at: last.updated_at, before_memory_id: last.memory_id }
        : undefined;
    },
  });
  const save = useMutation({
    mutationFn: ({ memory_id, body }: { memory_id: string; body: Capture }) =>
      fetchClient.PUT("/memory/v2/entries/{memory_id}", { params: { path: { memory_id } }, body }),
    onSuccess: () => {
      setEditing(null);
      toast.success("Memory updated");
      return cache.invalidateQueries({ queryKey: ["memoryEntries", userId] });
    },
    onError: (error: Error) => toast.error(error.message),
  });
  const remove = useMutation({
    mutationFn: (memory_id: string) =>
      fetchClient.DELETE("/memory/v2/entries/{memory_id}", { params: { path: { memory_id } } }),
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
    save.mutate({
      memory_id: editing.memory_id,
      body: {
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
      },
    });
  };
  const filtered = Boolean(search || teamId || filterUserId);
  const clearFilters = () => {
    setTeamId("");
    setFilterUserId("");
    setSearch("");
  };
  const accessDescription =
    "Your administrator controls saving and agent recall separately. These settings apply across your keys; existing permissions decide which memories you can see.";
  const gettingStarted = status.data?.save_enabled
    ? "Use your assistant as usual. Its saved memories will appear here."
    : "Your administrator can enable saving. Existing access permissions decide what you can see.";
  const moreLabel = entries.isError ? "Try again" : "Load more memories";
  return (
    <section className="space-y-6" aria-labelledby="memory-title">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-2">
          <h1 id="memory-title" className="text-[28px] font-semibold tracking-tight">
            Memory
          </h1>
          <p className="text-sm text-muted-foreground">
            What your assistants remember, with the people who contributed it.
          </p>
        </div>
        {status.isSuccess && (
          <span className="rounded-full border px-3 py-1 text-sm" role="status">
            Saving {status.data?.save_enabled ? "on" : "off"} · Agent recall {status.data?.read_enabled ? "on" : "off"}
          </span>
        )}
      </div>
      {status.error ? (
        <p role="alert" className="text-destructive">
          Could not load memory access: {status.error.message}
        </p>
      ) : (
        <>
          <p className="text-sm text-muted-foreground">
            {status.isPending ? "Loading memory access..." : accessDescription}
          </p>
          <div className="flex flex-wrap items-end gap-3">
            <div className="w-full space-y-1 sm:w-64">
              <Label htmlFor="memory-team-filter">Team</Label>
              <MemoryTeamPicker inputId="memory-team-filter" value={teamId} onChange={setTeamId} disabled={busy} />
            </div>
            {proxyAdmin && (
              <div className="w-full space-y-1 sm:w-64">
                <Label htmlFor="memory-author-filter">Contributor</Label>
                <MemoryUserPicker
                  inputId="memory-author-filter"
                  value={filterUserId}
                  onChange={setFilterUserId}
                  disabled={busy}
                />
              </div>
            )}
            {filtered && (
              <Button variant="ghost" onClick={clearFilters}>
                Clear filters
              </Button>
            )}
          </div>
          <MemoryEntriesTable
            entries={memories}
            query={search}
            onQueryChange={setSearch}
            loading={entries.isPending || status.isPending}
            readOnly={readOnly}
            busy={busy}
            onEdit={setEditing}
            onDelete={setDeleting}
            empty={
              entries.error ? (
                <p role="alert" className="text-destructive">
                  Could not load memories: {entries.error.message}
                </p>
              ) : (
                <div className="flex flex-col items-center gap-2 py-8 text-center">
                  <Brain className="mb-1 size-6 text-muted-foreground" />
                  <h3 className="font-medium">
                    {query || teamId || filterUserId ? "No matching memories" : "No memories yet"}
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {filtered ? "Try another search or clear the filters." : gettingStarted}
                  </p>
                </div>
              )
            }
            footer={
              <div className="flex items-center justify-between gap-3 px-4 py-3">
                <span className="text-xs text-muted-foreground">{memories.length} memories shown</span>
                {(entries.hasNextPage || entries.isError) && (
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={entries.isFetching}
                    onClick={() =>
                      entries.isError && !entries.isFetchNextPageError ? entries.refetch() : entries.fetchNextPage()
                    }
                  >
                    {entries.isFetching ? "Loading..." : moreLabel}
                  </Button>
                )}
              </div>
            }
          />
          {entries.isFetchNextPageError && (
            <p role="alert" className="text-destructive">
              Could not load more memories: {entries.error.message}
            </p>
          )}
        </>
      )}
      <details className="text-sm text-muted-foreground">
        <summary className="cursor-pointer">How access works</summary>
        <p className="mt-2">
          You can read your own memories and any teams&apos; memories you have permission to view. Team admins can
          inspect their team&apos;s records, and proxy admins can inspect all. Team permissions also apply when your
          assistant searches memory.
        </p>
      </details>
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
                <Button type="submit" disabled={busy}>
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
