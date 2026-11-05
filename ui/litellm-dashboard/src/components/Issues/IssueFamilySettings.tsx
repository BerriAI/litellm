import { useState, type FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCurrentUser } from "@/app/(dashboard)/hooks/currentUser/useCurrentUser";
import { getProxyBaseUrl } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { backgroundFetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { FAMILY_LABELS } from "./issueApi";

type Family = components["schemas"]["IssueFamilySetting"];
type Policy = components["schemas"]["IssueFamilyPolicy"];
type PolicyUpdate = components["schemas"]["IssueFamilyPolicyUpdate"];
type FamilyUpdate = components["schemas"]["IssueFamilyUpdate"];
const SELECT_CLASS = "h-9 rounded-md border border-input bg-background px-2 text-sm";
const FLOOR_UNITS = { spend: "USD", latency: "ms", reliability: "percentage points", traffic: "requests" };
const NUMBER_FIELDS = ["multiplier", "min_value", "min_requests", "baseline_days", "min_history_days"] as const;
function enabledLabel(enabled: boolean | null) {
  if (enabled === null) return "Mixed settings";
  return enabled ? "Enabled" : "Disabled";
}
function familySummary(family: Family) {
  if (family.total_rules === 0) return "No default policies yet";
  const thresholds = family.policies.map((policy) => {
    if (policy.multiplier === null) return "Thresholds vary by scope";
    return `${policy.multiplier}× ${policy.direction}`;
  });
  return [enabledLabel(family.enabled), ...new Set(thresholds)].join(" · ");
}
const enabledValue = (enabled: boolean | null) => (enabled === null ? "" : String(enabled));

function policyUpdate(data: FormData, policy: Policy): PolicyUpdate | null {
  const update: PolicyUpdate = { direction: policy.direction };
  for (const field of NUMBER_FIELDS) {
    const value = String(data.get(`${policy.direction}-${field}`) ?? "");
    if (value !== "" && Number(value) !== policy[field]) update[field] = Number(value);
  }
  const enabled = String(data.get(`${policy.direction}-enabled`) ?? "");
  if (enabled !== "" && enabled !== enabledValue(policy.enabled)) update.enabled = enabled === "true";
  return Object.keys(update).length > 1 ? update : null;
}

function FamilyEditor({
  family,
  pending,
  onSave,
  onCancel,
}: {
  family: Family;
  pending: boolean;
  onSave: (update: FamilyUpdate) => void;
  onCancel: () => void;
}) {
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const data = new FormData(event.currentTarget);
    const policies = family.policies.map((policy) => policyUpdate(data, policy)).filter((policy) => policy !== null);
    if (policies.length) onSave({ policies });
    else onCancel();
  };
  return (
    <form
      onSubmit={submit}
      className="space-y-4"
      onInvalidCapture={(event) => {
        if (event.target instanceof HTMLElement) {
          const details = event.target.closest("details");
          if (details) details.open = true;
        }
      }}
    >
      <h2 className="font-medium">{FAMILY_LABELS[family.family]}</h2>
      <p className="text-xs text-muted-foreground">
        Changes apply across the gateway, keys, teams, users, and models. Blank fields keep existing per-scope settings.
      </p>
      {family.policies.map((policy) => (
        <fieldset key={policy.direction} className="space-y-3 border-t pt-3">
          <legend className="pr-2 text-sm font-medium capitalize">{policy.direction}</legend>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="space-y-1">
              <Label htmlFor={`${policy.direction}-enabled`}>Monitoring</Label>
              <select
                id={`${policy.direction}-enabled`}
                name={`${policy.direction}-enabled`}
                defaultValue={enabledValue(policy.enabled)}
                className={`${SELECT_CLASS} w-full`}
              >
                <option value="">Keep mixed settings</option>
                <option value="true">Enabled</option>
                <option value="false">Disabled</option>
              </select>
            </div>
            <div className="space-y-1">
              <Label htmlFor={`${policy.direction}-multiplier`}>
                {policy.direction === "increase" ? "Increase factor" : "Decrease factor"}
              </Label>
              <Input
                id={`${policy.direction}-multiplier`}
                name={`${policy.direction}-multiplier`}
                type="number"
                min="1.01"
                max="1000"
                step="any"
                defaultValue={policy.multiplier ?? ""}
                placeholder="Varies by scope"
              />
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            {policy.direction === "increase"
              ? "A factor of 3 means at least three times the daily baseline."
              : "A factor of 5 means at most one fifth of the daily baseline."}
          </p>
          <details className="space-y-3">
            <summary className="cursor-pointer text-sm text-muted-foreground">Advanced</summary>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="space-y-1">
                <Label htmlFor={`${policy.direction}-min_value`}>Minimum change ({FLOOR_UNITS[family.family]})</Label>
                <Input
                  id={`${policy.direction}-min_value`}
                  name={`${policy.direction}-min_value`}
                  type="number"
                  min="0"
                  step="any"
                  defaultValue={policy.min_value ?? ""}
                  placeholder="Varies by scope"
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor={`${policy.direction}-min_requests`}>Minimum requests</Label>
                <Input
                  id={`${policy.direction}-min_requests`}
                  name={`${policy.direction}-min_requests`}
                  type="number"
                  min="0"
                  step="1"
                  defaultValue={policy.min_requests ?? ""}
                  placeholder="Varies by scope"
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor={`${policy.direction}-baseline_days`}>Baseline window (days)</Label>
                <Input
                  id={`${policy.direction}-baseline_days`}
                  name={`${policy.direction}-baseline_days`}
                  type="number"
                  min="1"
                  max="30"
                  step="1"
                  defaultValue={policy.baseline_days ?? ""}
                  placeholder="Varies by scope"
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor={`${policy.direction}-min_history_days`}>Required history (days)</Label>
                <Input
                  id={`${policy.direction}-min_history_days`}
                  name={`${policy.direction}-min_history_days`}
                  type="number"
                  min="1"
                  max="30"
                  step="1"
                  defaultValue={policy.min_history_days ?? ""}
                  placeholder="Varies by scope"
                />
              </div>
            </div>
          </details>
        </fieldset>
      ))}
      <div className="flex justify-end gap-2">
        <Button type="button" variant="outline" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit" disabled={pending}>
          {pending ? "Saving…" : "Save changes"}
        </Button>
      </div>
    </form>
  );
}

export default function IssueFamilySettings({ accessToken }: { accessToken: string | null }) {
  const { data: user } = useCurrentUser(accessToken);
  const [editing, setEditing] = useState<Family | null>(null);
  const client = useQueryClient();
  const options = {
    queryKey: ["issueFamilies", getProxyBaseUrl(), user?.user_id, user?.user_role],
    enabled: Boolean(accessToken && user?.user_id),
    queryFn: async ({ signal }: { signal: AbortSignal }) => {
      const { data } = await backgroundFetchClient.GET("/issues/families", { signal });
      if (!data) throw new Error("Could not load monitoring settings");
      return data;
    },
    retry: false,
  };
  const query = useQuery(options);
  const save = useMutation({
    mutationFn: async (body: FamilyUpdate) => {
      if (!editing) throw new Error("Choose an issue type");
      const { data } = await backgroundFetchClient.PATCH("/issues/families/{family}", {
        params: { path: { family: editing.family } },
        body,
      });
      if (!data) throw new Error("Could not save settings");
      return data;
    },
    onSuccess: async () => {
      setEditing(null);
      await client.invalidateQueries({ queryKey: ["issueFamilies"] });
    },
  });
  if (query.isPending)
    return (
      <p role="status" className="text-sm text-muted-foreground">
        Loading monitoring settings…
      </p>
    );
  if (query.isError)
    return (
      <div role="alert">
        <p>Could not load monitoring settings.</p>
        <Button variant="outline" onClick={() => void query.refetch()}>
          Retry
        </Button>
      </div>
    );
  return (
    <div className="space-y-3">
      {save.isError && (
        <p role="alert" className="text-sm text-destructive">
          {save.error.message}
        </p>
      )}
      {editing ? (
        <FamilyEditor
          key={editing.family}
          family={editing}
          pending={save.isPending}
          onSave={(body) => save.mutate(body)}
          onCancel={() => {
            setEditing(null);
            save.reset();
          }}
        />
      ) : (
        <>
          <p className="text-xs text-muted-foreground">
            Daily monitoring is enabled by default. Checks wait for a usable baseline and complete usage data. Model
            retirement issues appear automatically for used models.
          </p>
          <ul className="divide-y">
            {query.data.data.map((family) => (
              <li key={family.family} className="flex items-center justify-between gap-3 py-3">
                <div className="space-y-1">
                  <p className="text-sm font-medium">{FAMILY_LABELS[family.family]}</p>
                  <p className="text-xs text-muted-foreground">{familySummary(family)}</p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  aria-label={`Edit ${FAMILY_LABELS[family.family]}`}
                  onClick={() => {
                    save.reset();
                    setEditing(family);
                  }}
                >
                  Edit
                </Button>
              </li>
            ))}
          </ul>
          {query.data.data.length === 0 && (
            <p className="text-sm text-muted-foreground">No default performance policies are registered.</p>
          )}
        </>
      )}
    </div>
  );
}
