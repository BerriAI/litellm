"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { fetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { toast } from "@/lib/toast";
import { uiHref } from "@/utils/uiHref";
import { MemoryUserPicker } from "./MemoryTargetPicker";

type Settings = components["schemas"]["MemorySettings"];
type Enrollment = components["schemas"]["MemoryEnrollment"];
const OFF: Enrollment = { enabled: false, everyone: true, user_ids: [] };

function EnrollmentEditor({
  id,
  title,
  description,
  value,
  names,
  busy,
  onChange,
  onName,
}: Readonly<{
  id: string;
  title: string;
  description: string;
  value: Enrollment;
  names: Record<string, string>;
  busy: boolean;
  onChange: (value: Enrollment) => void;
  onName: (id: string, name: string) => void;
}>) {
  const addUser = (userId: string, name?: string) => {
    if (!userId) return;
    onChange({ ...value, user_ids: [...new Set([...(value.user_ids ?? []), userId])] });
    onName(userId, name ?? userId);
  };
  return (
    <div className="space-y-4 rounded-lg border p-5">
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <Label htmlFor={`${id}-enabled`} className="text-base">
            {title}
          </Label>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
        <Switch
          id={`${id}-enabled`}
          checked={value.enabled ?? false}
          disabled={busy}
          onCheckedChange={(enabled) => onChange({ ...value, enabled })}
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor={`${id}-enrollment`}>{id === "save" ? "Save for" : "Recall for"}</Label>
        <Select
          value={value.everyone ? "everyone" : "selected"}
          disabled={busy}
          onValueChange={(selection) => onChange({ ...value, everyone: selection === "everyone" })}
        >
          <SelectTrigger id={`${id}-enrollment`} className="w-full sm:max-w-sm">
            <SelectValue>{value.everyone ? "Everyone" : "Selected users"}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="everyone">Everyone</SelectItem>
            <SelectItem value="selected">Selected users</SelectItem>
          </SelectContent>
        </Select>
      </div>
      {!value.everyone && (
        <div className="space-y-3">
          <Label htmlFor={`${id}-user`}>Add a user to {id === "save" ? "saving" : "recall"}</Label>
          <MemoryUserPicker inputId={`${id}-user`} value="" disabled={busy} onChange={addUser} />
          <ul className="divide-y">
            {(value.user_ids ?? []).map((userId) => (
              <li key={userId} className="flex items-center justify-between gap-3 py-2">
                <span className="break-all text-sm">{names[userId] ?? userId}</span>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={busy}
                  aria-label={`Remove ${names[userId] ?? userId} from ${id === "save" ? "saving" : "recall"}`}
                  onClick={() => onChange({ ...value, user_ids: value.user_ids?.filter((item) => item !== userId) })}
                >
                  Remove
                </Button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

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
    queryFn: async ({ signal }) => (await fetchClient.GET("/memory/v2/settings", { signal })).data,
  });
  const current = draft ?? settings.data;
  const save = useMutation({
    mutationFn: ({ enabled, everyone, user_ids, read }: Settings) =>
      fetchClient.PUT("/memory/v2/settings", { body: { enabled, everyone, user_ids, read: read ?? OFF } }),
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
  return (
    <section className="space-y-6" aria-labelledby="memory-administration-title">
      <div>
        <h2 id="memory-administration-title" className="text-xl font-semibold">
          Administration
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Choose who can save memories and whose agents can use saved memories. Both are off by default.
        </p>
      </div>
      {proxyAdmin && settings.isPending && <p role="status">Loading memory settings...</p>}
      {settings.error && (
        <p role="alert" className="text-destructive">
          {settings.error.message}
        </p>
      )}
      {canShowSettings && current && (
        <div className="space-y-6">
          <EnrollmentEditor
            id="save"
            title="Save memories"
            description="Allow assistants to save new facts and decisions. Applies across each selected user’s keys."
            value={current}
            names={{ ...settings.data?.user_names, ...names }}
            busy={busy}
            onName={(id, name) => setNames((previous) => ({ ...previous, [id]: name }))}
            onChange={(value) => setDraft({ ...current, ...value })}
          />
          <EnrollmentEditor
            id="read"
            title="Use saved memories"
            description="Allow assistants to search and read existing memories under their current user and team permissions. This does not change dashboard or API access."
            value={current.read ?? OFF}
            names={{ ...settings.data?.user_names, ...names }}
            busy={busy}
            onName={(id, name) => setNames((previous) => ({ ...previous, [id]: name }))}
            onChange={(read) => setDraft({ ...current, read })}
          />
          <p className="text-sm text-muted-foreground">
            These controls are independent. Turn both off to stop automatic saving and recall. Existing memories remain
            available to authorized viewers. Enabling memory can add model calls, latency, and spend.
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
          all. To give ordinary members access to their team&apos;s memories, enable team memory access in Member
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
