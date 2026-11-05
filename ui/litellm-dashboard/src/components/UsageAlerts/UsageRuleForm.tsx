import { FormEvent, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { METRIC_LABELS, RuleInput, SCOPE_LABELS, UsageRule } from "./usageAlertApi";
import { DEFAULT_RULE, ruleInputFromForm } from "./ruleForm";

const METRIC_UNITS = { spend: "USD", latency: "ms", requests: "requests", errors: "errors" };

const SELECT_CLASS =
  "h-9 w-full rounded-md border border-input bg-background px-3 text-sm focus-visible:ring-2 focus-visible:ring-ring";

export default function UsageRuleForm({
  rule,
  pending,
  onSave,
  onCancel,
}: {
  rule: UsageRule | null;
  pending: boolean;
  onSave: (input: RuleInput) => void;
  onCancel: () => void;
}) {
  const initial = rule ?? DEFAULT_RULE;
  const [scope, setScope] = useState<RuleInput["scope_type"]>(initial.scope_type);
  const [metric, setMetric] = useState<RuleInput["metric"]>(initial.metric);
  const [baselineDays, setBaselineDays] = useState(initial.baseline_days ?? 7);
  const [historyDays, setHistoryDays] = useState(initial.min_history_days ?? 3);
  const changeBaselineDays = (value: string) => {
    const days = Number(value);
    setBaselineDays(days);
    if (days >= 1) setHistoryDays(Math.min(historyDays, days));
  };
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onSave(ruleInputFromForm(new FormData(event.currentTarget), scope, metric));
  };
  return (
    <form onSubmit={submit} className="space-y-4">
      <div className="space-y-1">
        <Label htmlFor="rule-name">Name</Label>
        <Input id="rule-name" name="name" required maxLength={200} defaultValue={initial.name} />
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1">
          <Label htmlFor="rule-scope">Scope</Label>
          <select
            id="rule-scope"
            className={SELECT_CLASS}
            value={scope}
            onChange={(event) => setScope(event.target.value as RuleInput["scope_type"])}
          >
            {Object.entries(SCOPE_LABELS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </div>
        <div className="space-y-1">
          <Label htmlFor="rule-scope-id">Entity ID</Label>
          <Input
            key={scope}
            id="rule-scope-id"
            name="scope_id"
            disabled={scope === "deployment"}
            defaultValue={scope === initial.scope_type ? initial.scope_id ?? "" : ""}
            placeholder={scope === "deployment" ? "Entire deployment" : "All entities"}
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="rule-metric">Metric</Label>
          <select
            id="rule-metric"
            className={SELECT_CLASS}
            value={metric}
            onChange={(event) => setMetric(event.target.value as RuleInput["metric"])}
          >
            {Object.entries(METRIC_LABELS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </div>
        <div className="space-y-1">
          <Label htmlFor="rule-direction">Change</Label>
          <select id="rule-direction" name="direction" className={SELECT_CLASS} defaultValue={initial.direction}>
            <option value="increase">Increase</option>
            <option value="decrease">Decrease</option>
          </select>
        </div>
        <div className="space-y-1">
          <Label htmlFor="rule-multiplier">Multiplier</Label>
          <Input
            id="rule-multiplier"
            name="multiplier"
            type="number"
            required
            min="1.01"
            max={1000}
            step="0.01"
            defaultValue={initial.multiplier}
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="rule-baseline">Baseline days</Label>
          <Input
            id="rule-baseline"
            name="baseline_days"
            type="number"
            required
            min={1}
            max={30}
            value={baselineDays}
            onChange={(event) => changeBaselineDays(event.target.value)}
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="rule-history">Minimum history days</Label>
          <Input
            id="rule-history"
            name="min_history_days"
            type="number"
            required
            min={1}
            max={baselineDays}
            value={historyDays}
            onChange={(event) => setHistoryDays(Number(event.target.value))}
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor="rule-requests">Minimum requests</Label>
          <Input
            id="rule-requests"
            name="min_requests"
            type="number"
            required
            min={0}
            defaultValue={initial.min_requests}
          />
        </div>
        <div className="space-y-1 col-span-2">
          <Label htmlFor="rule-minimum">Minimum absolute change ({METRIC_UNITS[metric]})</Label>
          <Input
            id="rule-minimum"
            name="min_value"
            type="number"
            required
            min={0}
            step="any"
            defaultValue={initial.min_value}
          />
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        Compares each completed UTC day with the average of recorded days in the baseline window. A 3× decrease means
        falling below one-third of the baseline.
      </p>
      <div className="flex gap-5">
        <label className="flex items-center gap-2">
          <input type="checkbox" name="email" defaultChecked={initial.notification_channels?.includes("email")} /> Email
        </label>
        <label className="flex items-center gap-2">
          <input type="checkbox" name="slack" defaultChecked={initial.notification_channels?.includes("slack")} /> Slack
        </label>
      </div>
      <div className="space-y-1">
        <Label htmlFor="rule-emails">Email recipients</Label>
        <Input
          id="rule-emails"
          name="email_recipients"
          defaultValue={initial.email_recipients?.join(", ") ?? ""}
          placeholder="Separate addresses with commas"
        />
      </div>
      <p className="text-xs text-muted-foreground">Uses the gateway’s configured email and Slack connections.</p>
      <label className="flex items-center gap-2">
        <input type="checkbox" name="enabled" defaultChecked={initial.enabled} /> Enabled
      </label>
      <div className="flex justify-end gap-2">
        <Button type="button" variant="outline" onClick={onCancel} disabled={pending}>
          Cancel
        </Button>
        <Button type="submit" disabled={pending}>
          {pending ? "Saving…" : "Save rule"}
        </Button>
      </div>
    </form>
  );
}
