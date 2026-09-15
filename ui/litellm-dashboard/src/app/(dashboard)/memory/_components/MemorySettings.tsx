"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { fetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { toast } from "@/lib/toast";
import { uiHref } from "@/utils/uiHref";
import { MemoryUserPicker } from "./MemoryTargetPicker";

type Settings = components["schemas"]["MemorySettings"];

export function MemoryAdministration({
  userId,
  proxyAdmin,
  readOnly,
}: Readonly<{ userId: string; proxyAdmin: boolean; readOnly: boolean }>) {
  const cache = useQueryClient();
  const [draft, setDraft] = useState<Settings | null>(null);
  const [names, setNames] = useState<Record<string, string>>({});
  const settings = useQuery({
    queryKey: ["memorySettings", userId],
    enabled: proxyAdmin,
    queryFn: async ({ signal }) => (await fetchClient.GET("/v2/memory/settings", { signal })).data,
  });
  const current = draft ?? settings.data;
  const save = useMutation({
    mutationFn: ({ enabled, everyone, user_ids }: Settings) =>
      fetchClient.PUT("/v2/memory/settings", { body: { enabled, everyone, user_ids } }),
    onSuccess: async ({ data }) => {
      cache.setQueryData(["memorySettings", userId], data);
      setDraft(null);
      toast.success("Memory settings saved");
      await Promise.all([
        cache.invalidateQueries({ queryKey: ["memoryStatus"], refetchType: "all" }),
        cache.invalidateQueries({ queryKey: ["memoryEntries"] }),
      ]);
    },
  });
  const busy = readOnly || save.isPending;
  const canShowSettings = Boolean(proxyAdmin && current && !settings.error);
  const addUser = (id: string, label?: string) => {
    if (!id || !current) return;
    setDraft({ ...current, user_ids: [...new Set([...(current.user_ids ?? []), id])] });
    setNames((previous) => ({ ...previous, [id]: label ?? id }));
  };
  return (
    <section className="space-y-6" aria-labelledby="memory-administration-title">
      <div>
        <h2 id="memory-administration-title" className="text-xl font-semibold">
          Administration
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">Enable memory for the people who should use it.</p>
      </div>
      {proxyAdmin && settings.isPending && <p role="status">Loading memory settings...</p>}
      {settings.error && (
        <p role="alert" className="text-destructive">
          {settings.error.message}
        </p>
      )}
      {canShowSettings && current && (
        <div className="space-y-6 rounded-lg border p-5">
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-1">
              <Label htmlFor="memory-enabled" className="text-base">
                Gateway memory
              </Label>
              <p className="text-sm text-muted-foreground">
                Off by default. When enabled, assistants can save and recall memories through the gateway.
              </p>
            </div>
            <Switch
              id="memory-enabled"
              checked={current.enabled ?? false}
              disabled={busy}
              onCheckedChange={(enabled) => setDraft({ ...current, enabled })}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="memory-enrollment">Enable for</Label>
            <select
              id="memory-enrollment"
              className="h-9 w-full rounded-md border bg-background px-3 text-sm sm:max-w-sm"
              value={current.everyone ? "everyone" : "selected"}
              disabled={busy}
              onChange={(event) => setDraft({ ...current, everyone: event.target.value === "everyone" })}
            >
              <option value="everyone">Everyone</option>
              <option value="selected">Selected users</option>
            </select>
          </div>
          {!current.everyone && (
            <div className="space-y-3">
              <Label htmlFor="memory-enrolled-user">Add a user</Label>
              <MemoryUserPicker inputId="memory-enrolled-user" value="" disabled={busy} onChange={addUser} />
              <ul className="divide-y">
                {(current.user_ids ?? []).map((id) => (
                  <li key={id} className="flex items-center justify-between gap-3 py-2">
                    <span className="break-all text-sm">{names[id] ?? settings.data?.user_names?.[id] ?? id}</span>
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={busy}
                      aria-label={`Remove ${names[id] ?? settings.data?.user_names?.[id] ?? id}`}
                      onClick={() =>
                        setDraft({ ...current, user_ids: current.user_ids?.filter((value) => value !== id) })
                      }
                    >
                      Remove
                    </Button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <p className="text-sm text-muted-foreground">
            Turning memory off stops automatic saving and recall. Existing memories remain available to authorized
            viewers. Enabling memory can add model calls, latency, and spend.
          </p>
          {save.error && (
            <p role="alert" className="text-destructive">
              {save.error.message}
            </p>
          )}
          <div className="flex items-center justify-end gap-3">
            {draft && <span className="text-sm text-muted-foreground">Unsaved changes</span>}
            <Button
              variant="outline"
              disabled={busy || !draft}
              onClick={() => {
                setDraft(null);
                save.reset();
              }}
            >
              Reset
            </Button>
            <Button disabled={busy || !draft} onClick={() => save.mutate(current)}>
              Save changes
            </Button>
          </div>
        </div>
      )}
      <div className="space-y-2 rounded-lg border p-5">
        <h3 className="font-medium">Who can see memories?</h3>
        <p className="text-sm text-muted-foreground">
          Users see their own memories. Team admins can also see their team&apos;s memories, and proxy admins can see
          all. To give ordinary members access to their team&apos;s memories, allow “Read team memories” in Member
          Permissions.
        </p>
        <Link className="inline-block text-sm underline underline-offset-4" href={uiHref("teams")}>
          Manage team permissions
        </Link>
      </div>
      <p className="text-xs text-muted-foreground">These settings do not change Memory API (V1).</p>
    </section>
  );
}
