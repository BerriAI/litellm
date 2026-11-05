import { FormEvent, useState } from "react";
import { ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { METRIC_LABELS, RuleInput, SCOPE_LABELS, UsageRule } from "./usageAlertApi";
import { DEFAULT_RULE, METRIC_MINIMUMS, ruleInputFromForm } from "./ruleForm";
import UsageRuleResourceSelect, { allResourcesLabel } from "./UsageRuleResourceSelect";

const METRIC_UNITS = { spend: "USD", latency: "ms", requests: "requests", errors: "errors" };

const SELECT_CLASS =
  "h-9 w-full rounded-md border border-input bg-background px-3 text-sm focus-visible:ring-2 focus-visible:ring-ring";

const initialResource = (rule: RuleInput) => ({
  id: rule.scope_id ?? null,
  label: rule.scope_id || allResourcesLabel(rule.scope_type),
});

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
  const builtIn = rule?.is_default === true;
  const [scope, setScope] = useState<RuleInput["scope_type"]>(initial.scope_type);
  const [metric, setMetric] = useState<RuleInput["metric"]>(initial.metric);
  const [resource, setResource] = useState(() => initialResource(initial));
  const [direction, setDirection] = useState(initial.direction);
  const [name, setName] = useState<string | null>(rule?.name ?? null);
  const [minimum, setMinimum] = useState(initial.min_value ?? METRIC_MINIMUMS[initial.metric]);
  const [minimumChanged, setMinimumChanged] = useState(false);
  const [email, setEmail] = useState(initial.notification_channels?.includes("email") ?? false);
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
  const generatedName = `${METRIC_LABELS[metric]} ${direction} for ${resource.label}`.slice(0, 200);
  const changeScope = (next: RuleInput["scope_type"]) => {
    setScope(next);
    setResource({ id: null, label: allResourcesLabel(next) });
  };
  const changeMetric = (next: RuleInput["metric"]) => {
    if (!rule && !minimumChanged) setMinimum(METRIC_MINIMUMS[next]);
    setMetric(next);
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
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-1">
          <Label htmlFor="rule-scope">Resource</Label>
          <select
            id="rule-scope"
            className={SELECT_CLASS}
            value={scope}
            disabled={builtIn}
            onChange={(event) => changeScope(event.target.value as RuleInput["scope_type"])}
          >
            {Object.entries(SCOPE_LABELS).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </div>
        {scope !== "deployment" && (
          <div className="min-w-0 space-y-1">
            <Label htmlFor="rule-resource">{SCOPE_LABELS[scope]}</Label>
            {builtIn ? (
              <Input id="rule-resource" value={allResourcesLabel(scope)} disabled />
            ) : (
              <UsageRuleResourceSelect key={scope} scope={scope} value={resource.id} onChange={setResource} />
            )}
          </div>
        )}
        <input type="hidden" name="scope_id" value={resource.id ?? ""} />
      </div>
      <div className="grid grid-cols-2 items-end gap-4 sm:grid-cols-3">
        <div className="col-span-2 space-y-1 sm:col-span-1">
          <Label htmlFor="rule-metric">Metric</Label>
          <select
            id="rule-metric"
            className={SELECT_CLASS}
            value={metric}
            disabled={builtIn}
            onChange={(event) => changeMetric(event.target.value as RuleInput["metric"])}
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
          <select
            id="rule-direction"
            className={SELECT_CLASS}
            value={direction}
            disabled={builtIn}
            onChange={(event) => setDirection(event.target.value as RuleInput["direction"])}
          >
            <option value="increase">Increase</option>
            <option value="decrease">Decrease</option>
          </select>
          <input type="hidden" name="direction" value={direction} />
        </div>
        <div className="space-y-1">
          <Label htmlFor="rule-multiplier">Threshold (×)</Label>
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
      </div>
      <p className="text-xs leading-relaxed text-muted-foreground">
        Compares completed UTC days with prior usage. A 3× decrease means one-third of the baseline or lower.
      </p>
      {builtIn && (
        <p className="text-xs text-muted-foreground">Use a custom rule to change the resource, metric, or direction.</p>
      )}
      <div className="space-y-3 border-t pt-4">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
          <span className="text-muted-foreground">Also notify</span>
          <label className="flex items-center gap-2">
            <input type="checkbox" name="email" checked={email} onChange={(event) => setEmail(event.target.checked)} />{" "}
            Email
          </label>
          <label className="flex items-center gap-2">
            <input type="checkbox" name="slack" defaultChecked={initial.notification_channels?.includes("slack")} />{" "}
            Slack
          </label>
        </div>
        <div className={email ? "space-y-1" : "hidden"}>
          <Label htmlFor="rule-emails">Email recipients</Label>
          <Input
            id="rule-emails"
            name="email_recipients"
            required={email}
            disabled={!email}
            defaultValue={initial.email_recipients?.join(", ") ?? ""}
            placeholder="Separate addresses with commas"
          />
        </div>
        <p className="text-xs text-muted-foreground">Uses the gateway’s email and Slack connections.</p>
      </div>
      <details className="group border-t pt-3">
        <summary className="flex cursor-pointer list-none items-center gap-1 text-sm">
          <ChevronRight className="size-4 text-muted-foreground transition-transform group-open:rotate-90" />
          Advanced
        </summary>
        <div className="grid grid-cols-2 items-end gap-4 pt-4">
          <div className="col-span-2 space-y-1">
            <Label htmlFor="rule-name">Name</Label>
            <Input
              id="rule-name"
              name="name"
              required
              maxLength={200}
              value={name ?? generatedName}
              onChange={(event) => setName(event.target.value)}
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
          <div className="space-y-1">
            <Label htmlFor="rule-minimum">Minimum change ({METRIC_UNITS[metric]})</Label>
            <Input
              id="rule-minimum"
              name="min_value"
              type="number"
              required
              min={0}
              step="any"
              value={minimum}
              onChange={(event) => {
                setMinimumChanged(true);
                setMinimum(Number(event.target.value));
              }}
            />
          </div>
          <p className="col-span-2 text-xs leading-relaxed text-muted-foreground">
            The baseline uses recorded days in this window. Minimum changes and request counts reduce alerts from small
            samples.
          </p>
          <label className="col-span-2 flex items-center gap-2 text-sm">
            <input type="checkbox" name="enabled" defaultChecked={initial.enabled} /> Enabled
          </label>
        </div>
      </details>
      <div className="flex justify-end gap-2 border-t pt-4">
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
