"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { fetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { toast } from "@/lib/toast";

import { MemoryTargetPicker } from "./MemoryTargetPicker";

type PolicyInput = components["schemas"]["MemoryPolicyInput"];
type Policy = components["schemas"]["MemoryPolicy"];
const activationNames = {
  disabled: "Disabled",
  opt_in: "Users choose whether to opt in",
  automatic: "Enabled automatically",
} as const;
const scopeNames = {
  key: "Private to each virtual key",
  user: "Private to each user within an organization",
  team: "Shared within the team",
  project: "Shared within the project",
  organization: "Shared within the organization",
} as const;
const targetNames = {
  gateway: "Whole gateway",
  organization: "Organization",
  team: "Team",
  project: "Project",
  user: "User",
  key: "Virtual key",
} as const;
const selectClass = "h-9 w-full rounded-md border bg-background px-3 text-sm";

export function MemoryPreference({
  userId,
  keyId,
  status,
  readOnly,
}: Readonly<{ userId: string; keyId: string; status: components["schemas"]["MemoryStatus"]; readOnly: boolean }>) {
  const cache = useQueryClient();
  const save = useMutation({
    mutationFn: async (enabled: boolean) =>
      fetchClient.PUT("/v2/memory/preference", { params: { query: { key_id: keyId } }, body: { enabled } }),
    onSettled: () =>
      Promise.all([
        cache.invalidateQueries({ queryKey: ["memoryStatus", userId] }),
        cache.invalidateQueries({ queryKey: ["memoryEntries", userId] }),
      ]),
    onError: (error: Error) => toast.error(error.message),
  });
  const canToggle = status.activation === "opt_in" && !!status.scope;
  return (
    <div className="flex items-center gap-3 rounded-lg border bg-card px-4 py-3">
      <Label htmlFor="memory-enabled" className="font-medium">
        Memory <span aria-hidden="true">{status.active ? "on" : "off"}</span>
      </Label>
      <Switch
        id="memory-enabled"
        aria-label="Memory"
        checked={status.active}
        disabled={readOnly || !canToggle || save.isPending}
        onCheckedChange={(enabled) => save.mutate(enabled)}
      />
      {save.isPending && (
        <span role="status" className="sr-only">
          Updating memory
        </span>
      )}
    </div>
  );
}

export function MemoryPolicies({
  userId,
  proxyAdmin,
  readOnly,
}: Readonly<{ userId: string; proxyAdmin: boolean; readOnly: boolean }>) {
  const cache = useQueryClient();
  const initialPolicy: PolicyInput = {
    target_type: proxyAdmin ? "gateway" : "team",
    target_id: proxyAdmin ? "*" : "",
    activation: "opt_in",
    scope: "key",
  };
  const [policy, setPolicy] = useState<PolicyInput>(initialPolicy);
  const [offset, setOffset] = useState(0);
  const changeTarget = (target: PolicyInput["target_type"]) => {
    const selection: PolicyInput = {
      ...policy,
      target_type: target,
      target_id: target === "gateway" ? "*" : "",
      scope: "key",
    };
    setPolicy(selection);
    setOffset(0);
  };
  const filters = proxyAdmin ? { offset } : { target_type: policy.target_type, target_id: policy.target_id, offset };
  const allowedScope = (scope: string) => {
    if (proxyAdmin) return true;
    if (scope === "user") return false;
    return scope !== "organization" || policy.target_type === "organization";
  };
  const queryKey = ["memoryPolicies", userId, filters];
  const policies = useQuery({
    queryKey,
    queryFn: async ({ signal }) =>
      (await fetchClient.GET("/v2/memory/policies", { params: { query: filters }, signal })).data,
    enabled: proxyAdmin || !!policy.target_id,
    retry: false,
  });
  const invalidate = () =>
    Promise.all([
      cache.invalidateQueries({ queryKey: ["memoryPolicies", userId] }),
      cache.invalidateQueries({ queryKey: ["memoryStatus"] }),
      cache.invalidateQueries({ queryKey: ["memoryEntries"] }),
    ]);
  const save = useMutation({
    mutationFn: async (body: PolicyInput) => fetchClient.PUT("/v2/memory/policies", { body }),
    onSuccess: () => {
      toast.success("Memory policy saved");
      return invalidate();
    },
    onError: (error: Error) => toast.error(error.message),
  });
  const remove = useMutation({
    mutationFn: async (policy_id: string) =>
      fetchClient.DELETE("/v2/memory/policies/{policy_id}", { params: { path: { policy_id } } }),
    onSuccess: () => {
      toast.success("Memory policy removed; inherited settings now apply");
      return invalidate();
    },
    onError: (error: Error) => toast.error(error.message),
  });
  const busy = readOnly || save.isPending || remove.isPending;
  const edit = (row: Policy) => {
    const selected: PolicyInput = {
      target_type: row.target_type,
      target_id: row.target_id,
      activation: row.activation,
      scope: row.scope,
    };
    setPolicy(selected);
  };
  return (
    <section className="rounded-lg border p-5 space-y-4" aria-labelledby="memory-policy-title">
      <div>
        <h2 id="memory-policy-title" className="font-semibold">
          Memory policies
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Choose who can use memory and how it is shared. Users opt in by default. Enabling memory can add model calls,
          latency, and spend.
        </p>
      </div>
      <form
        className="grid gap-4 md:grid-cols-2"
        onSubmit={(event) => {
          event.preventDefault();
          save.mutate(policy);
        }}
      >
        <div className="space-y-2">
          <Label htmlFor="memory-target-type">Apply to</Label>
          <select
            id="memory-target-type"
            className={selectClass}
            value={policy.target_type}
            disabled={busy}
            onChange={(event) => changeTarget(event.target.value as PolicyInput["target_type"])}
          >
            {Object.entries(targetNames)
              .filter(([target]) => proxyAdmin || (target !== "gateway" && target !== "user"))
              .map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
          </select>
        </div>
        {policy.target_type !== "gateway" && (
          <div className="space-y-2">
            <Label htmlFor="memory-target">{targetNames[policy.target_type]}</Label>
            <MemoryTargetPicker
              key={policy.target_type}
              target={policy.target_type}
              value={policy.target_id}
              disabled={busy}
              onChange={(target_id) => {
                setPolicy({ ...policy, target_id });
                setOffset(0);
              }}
            />
          </div>
        )}
        <div className="space-y-2">
          <Label htmlFor="memory-activation">Activation</Label>
          <select
            id="memory-activation"
            className={selectClass}
            value={policy.activation}
            disabled={busy}
            onChange={(event) => setPolicy({ ...policy, activation: event.target.value as PolicyInput["activation"] })}
          >
            {Object.entries(activationNames).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </div>
        <div className="space-y-2">
          <Label htmlFor="memory-scope">Who shares the memories</Label>
          <select
            id="memory-scope"
            className={selectClass}
            value={policy.scope}
            disabled={busy}
            onChange={(event) => setPolicy({ ...policy, scope: event.target.value as PolicyInput["scope"] })}
          >
            {Object.entries(scopeNames)
              .filter(([scope]) => allowedScope(scope))
              .map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
          </select>
        </div>
        <div className="md:col-span-2 text-sm text-muted-foreground">
          More specific policies take precedence: virtual key, user, project, team, organization, then gateway. Changing
          the sharing scope starts using that scope&apos;s memories; existing entries remain stored.
        </div>
        {!readOnly && (
          <Button type="submit" disabled={busy || !policy.target_id}>
            {save.isPending ? "Saving..." : "Save memory policy"}
          </Button>
        )}
      </form>
      {policies.error && (
        <p role="alert" className="text-sm text-destructive">
          {policies.error.message}
        </p>
      )}
      {policies.isLoading && <p role="status">Loading memory policies...</p>}
      {policies.data?.length === 0 && (
        <p className="text-sm text-muted-foreground">No policies set for this selection</p>
      )}
      <ul className="divide-y">
        {(policies.data ?? []).map((row) => (
          <li key={row.policy_id} className="flex flex-wrap items-center justify-between gap-3 py-3">
            <div className="min-w-0">
              <p className="font-medium">
                {targetNames[row.target_type]}: {row.target_id === "*" ? "All requests" : row.target_id}
              </p>
              <p className="text-sm text-muted-foreground">
                {activationNames[row.activation]} · {scopeNames[row.scope ?? "key"]}
              </p>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" disabled={busy} onClick={() => edit(row)}>
                Edit
              </Button>
              <Button variant="outline" disabled={busy} onClick={() => remove.mutate(row.policy_id)}>
                Use inherited policy
              </Button>
            </div>
          </li>
        ))}
      </ul>
      {(offset > 0 || policies.data?.length === 100) && (
        <div className="flex gap-2">
          <Button
            variant="outline"
            disabled={offset === 0 || policies.isFetching}
            onClick={() => setOffset(Math.max(0, offset - 100))}
          >
            Previous policies
          </Button>
          <Button
            variant="outline"
            disabled={policies.data?.length !== 100 || policies.isFetching}
            onClick={() => setOffset(offset + 100)}
          >
            More policies
          </Button>
        </div>
      )}
    </section>
  );
}
