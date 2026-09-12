"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { fetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { toast } from "@/lib/toast";

import { MemoryKeyPicker } from "./MemoryTargetPicker";

type Entry = components["schemas"]["MemoryEntry"];
type Capture = components["schemas"]["MemoryCapture"];

export function AutomaticMemoryEntries({ userId, readOnly }: Readonly<{ userId: string; readOnly: boolean }>) {
  const cache = useQueryClient();
  const [keyId, setKeyId] = useState("");
  const [query, setQuery] = useState("");
  const [offset, setOffset] = useState(0);
  const [editing, setEditing] = useState<Entry | null>(null);
  const [deleting, setDeleting] = useState<Entry | null>(null);
  const status = useQuery({
    queryKey: ["memoryStatus", userId, keyId],
    enabled: !!keyId,
    queryFn: async ({ signal }) =>
      (await fetchClient.GET("/v2/memory/status", { params: { query: { key_id: keyId } }, signal })).data,
  });
  const entries = useQuery({
    queryKey: ["memoryEntries", userId, keyId, query, offset],
    enabled: !!keyId && !!status.data?.scope,
    queryFn: async ({ signal }) =>
      (
        await fetchClient.GET("/v2/memory/entries", {
          params: { query: { key_id: keyId, query, offset, limit: 20 } },
          signal,
        })
      ).data,
  });
  const save = useMutation({
    mutationFn: async ({ key, body }: { key: string; body: Capture }) =>
      fetchClient.POST("/v2/memory/entries", { params: { query: { key_id: key } }, body }),
    onSuccess: (_, variables) => {
      setEditing(null);
      toast.success("Memory updated");
      return cache.invalidateQueries({ queryKey: ["memoryEntries", userId, variables.key] });
    },
    onError: (error: Error) => toast.error(error.message),
  });
  const remove = useMutation({
    mutationFn: async ({ key, memory_id }: { key: string; memory_id: string }) =>
      fetchClient.DELETE("/v2/memory/entries/{memory_id}", { params: { path: { memory_id }, query: { key_id: key } } }),
    onSuccess: (_, variables) => {
      setDeleting(null);
      toast.success("Memory deleted");
      return cache.invalidateQueries({ queryKey: ["memoryEntries", userId, variables.key] });
    },
    onError: (error: Error) => toast.error(error.message),
  });
  const busy = save.isPending || remove.isPending;
  const selectKey = (value: string) => {
    setKeyId(value);
    setOffset(0);
    setEditing(null);
    setDeleting(null);
  };
  return (
    <section className="rounded-lg border p-5 space-y-4" aria-labelledby="automatic-entries-title">
      <h2 id="automatic-entries-title" className="font-semibold">
        Saved gateway memories
      </h2>
      <div className="max-w-lg space-y-2">
        <Label htmlFor="memory-entry-key">Virtual key</Label>
        <MemoryKeyPicker inputId="memory-entry-key" value={keyId} disabled={busy} onChange={selectKey} />
      </div>
      {status.data && (
        <p className="text-sm text-muted-foreground">
          {status.data.active ? "Automatic memory is active" : "Automatic memory is off"}
          {status.data.scope ? ` · ${status.data.scope} scope` : " · No applicable policy"}
        </p>
      )}
      {(status.error || entries.error) && (
        <p role="alert" className="text-sm text-destructive">
          {status.error?.message || entries.error?.message}
        </p>
      )}
      {status.data?.scope && (
        <Input
          aria-label="Search saved memories"
          placeholder="Search memory text or keys"
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setOffset(0);
          }}
        />
      )}
      {entries.isFetching && <p role="status">Loading memories...</p>}
      {entries.data?.length === 0 && <p className="text-sm text-muted-foreground">No memories match this search</p>}
      <ul className="divide-y">
        {(entries.data ?? []).map((entry) => (
          <li key={entry.memory_id} className="space-y-2 py-4">
            <h3 className="font-medium">{entry.title}</h3>
            <p className="whitespace-pre-wrap text-sm">{entry.content}</p>
            <p className="text-xs text-muted-foreground">Evidence: {entry.evidence}</p>
            {!readOnly && (
              <div className="flex gap-2">
                <Button variant="outline" disabled={busy || !status.data?.active} onClick={() => setEditing(entry)}>
                  Edit memory
                </Button>
                <Button variant="outline" disabled={busy} onClick={() => setDeleting(entry)}>
                  Delete memory
                </Button>
              </div>
            )}
          </li>
        ))}
      </ul>
      {editing && (
        <form
          className="space-y-3 rounded-md border p-4"
          onSubmit={(event) => {
            event.preventDefault();
            save.mutate({
              key: keyId,
              body: {
                key: editing.key,
                title: editing.title,
                content: editing.content,
                evidence: editing.evidence,
                expected_revision: editing.updated_at,
              },
            });
          }}
        >
          <Label htmlFor="memory-edit-content">Correct this memory</Label>
          <Textarea
            id="memory-edit-content"
            value={editing.content}
            required
            maxLength={8000}
            disabled={busy}
            onChange={(event) => setEditing({ ...editing, content: event.target.value })}
          />
          <Label htmlFor="memory-edit-evidence">Evidence</Label>
          <Textarea
            id="memory-edit-evidence"
            value={editing.evidence}
            required
            maxLength={2000}
            disabled={busy}
            onChange={(event) => setEditing({ ...editing, evidence: event.target.value })}
          />
          <div className="flex gap-2">
            <Button type="submit" disabled={busy || !status.data?.active}>
              Save correction
            </Button>
            <Button type="button" variant="outline" disabled={busy} onClick={() => setEditing(null)}>
              Cancel
            </Button>
          </div>
        </form>
      )}
      {(offset > 0 || entries.data?.length === 20) && (
        <div className="flex gap-2">
          <Button
            variant="outline"
            disabled={offset === 0 || entries.isFetching}
            onClick={() => setOffset(Math.max(0, offset - 20))}
          >
            Previous memories
          </Button>
          <Button
            variant="outline"
            disabled={entries.data?.length !== 20 || entries.isFetching}
            onClick={() => setOffset(offset + 20)}
          >
            More memories
          </Button>
        </div>
      )}
      <DeleteResourceModal
        isOpen={deleting !== null}
        onCancel={() => setDeleting(null)}
        onOk={() => {
          if (deleting) remove.mutate({ key: keyId, memory_id: deleting.memory_id });
        }}
        title="Delete memory"
        message={`Delete ${deleting?.title ?? "this memory"}?`}
        confirmLoading={remove.isPending}
      />
    </section>
  );
}
