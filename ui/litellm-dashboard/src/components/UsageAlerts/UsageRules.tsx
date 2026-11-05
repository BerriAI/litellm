import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { backgroundFetchClient } from "@/lib/http/api";
import { useCurrentUser } from "@/app/(dashboard)/hooks/currentUser/useCurrentUser";
import { getProxyBaseUrl } from "@/components/networking";
import { METRIC_LABELS, RuleInput, RulePage, SCOPE_LABELS, UsageRule } from "./usageAlertApi";
import UsageRuleForm from "./UsageRuleForm";

const RULE_VIEWS = {
  custom: {
    title: "Custom rules",
    switchLabel: "View defaults",
    empty: "No custom rules. Add one to change a threshold or notify your team.",
  },
  default: { title: "Default rules", switchLabel: "View custom rules", empty: "No default rules registered." },
};

function editorTitle(rule: UsageRule | null | undefined): string {
  if (rule === undefined) return "Alert rules";
  return rule ? "Edit rule" : "New rule";
}

export default function UsageRules({ accessToken }: { accessToken: string | null }) {
  const { data: user } = useCurrentUser(accessToken);
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState<UsageRule | null | undefined>(undefined);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [kind, setKind] = useState<"custom" | "default">("custom");
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: ["usageAlertRules", getProxyBaseUrl(), user?.user_id, user?.user_role, offset, kind],
    enabled: open && Boolean(accessToken && user?.user_id),
    queryFn: async ({ signal }) => {
      const { data } = await backgroundFetchClient.GET("/usage/alert-rules", {
        params: { query: { limit: 20, offset, kind } },
        signal,
      });
      if (!data) throw new Error("Could not load rules");
      return data;
    },
  });
  const save = useMutation({
    mutationFn: async (body: RuleInput) =>
      editing
        ? backgroundFetchClient.PUT("/usage/alert-rules/{rule_id}", { params: { path: { rule_id: editing.id } }, body })
        : backgroundFetchClient.POST("/usage/alert-rules", { body }),
    onSuccess: async () => {
      setEditing(undefined);
      await queryClient.invalidateQueries({ queryKey: ["usageAlertRules"] });
    },
  });
  const remove = useMutation({
    mutationFn: (id: string) =>
      backgroundFetchClient.DELETE("/usage/alert-rules/{rule_id}", { params: { path: { rule_id: id } } }),
    onSuccess: async () => {
      setDeleting(null);
      if (query.data?.data.length === 1) setOffset(Math.max(0, offset - 20));
      await queryClient.invalidateQueries({ queryKey: ["usageAlertRules"] });
    },
  });
  const error = save.error ?? remove.error;
  const total = query.data?.total ?? 0;
  const view = RULE_VIEWS[kind];
  const changeKind = () => {
    setKind(kind === "custom" ? "default" : "custom");
    setOffset(0);
    setDeleting(null);
  };
  return (
    <Dialog
      open={open}
      onOpenChange={(value) => {
        setOpen(value);
        if (!value) {
          setEditing(undefined);
          setDeleting(null);
          save.reset();
          remove.reset();
        }
      }}
    >
      <DialogTrigger render={<Button variant="outline" size="sm" />}>Alert rules</DialogTrigger>
      <DialogContent className="sm:max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{editorTitle(editing)}</DialogTitle>
          <DialogDescription>Set the changes that should trigger an alert.</DialogDescription>
        </DialogHeader>
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error.message}
          </p>
        )}
        {editing !== undefined ? (
          <UsageRuleForm
            key={editing?.id ?? "new"}
            rule={editing}
            pending={save.isPending}
            onSave={(input) => save.mutate(input)}
            onCancel={() => {
              setEditing(undefined);
              save.reset();
            }}
          />
        ) : (
          <>
            <DefaultRuleSummary summary={query.data?.default_summary} />
            <div className="flex flex-wrap items-center justify-between gap-2 border-t pt-3">
              <div className="flex items-center gap-3">
                <p className="text-sm font-medium">{view.title}</p>
                <Button variant="link" size="sm" className="h-auto p-0 text-xs" onClick={changeKind}>
                  {view.switchLabel}
                </Button>
              </div>
              <Button size="sm" onClick={() => setEditing(null)}>
                New rule
              </Button>
            </div>
            {query.isPending && <p>Loading rules…</p>}
            {query.isError && (
              <div role="alert">
                <p>Could not load rules.</p>
                <Button variant="outline" onClick={() => void query.refetch()}>
                  Retry
                </Button>
              </div>
            )}
            {query.isSuccess && query.data.data.length === 0 && (
              <p className="py-4 text-sm text-muted-foreground">{view.empty}</p>
            )}
            {query.isSuccess && (
              <ul className="min-w-0 divide-y">
                {query.data.data.map((rule) => (
                  <RuleRow
                    key={rule.id}
                    rule={rule}
                    confirming={deleting === rule.id}
                    pending={remove.isPending}
                    onEdit={() => {
                      save.reset();
                      setEditing(rule);
                    }}
                    onDelete={() => {
                      if (deleting === rule.id) remove.mutate(rule.id);
                      else setDeleting(rule.id);
                    }}
                    onCancel={() => setDeleting(null)}
                  />
                ))}
              </ul>
            )}
            {total > 20 && (
              <div className="flex items-center justify-between">
                <span className="text-xs text-muted-foreground">
                  {total} {kind} rules
                </span>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={offset === 0}
                    onClick={() => setOffset(Math.max(0, offset - 20))}
                  >
                    Previous
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={offset + 20 >= total}
                    onClick={() => setOffset(offset + 20)}
                  >
                    Next
                  </Button>
                </div>
              </div>
            )}
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

function DefaultRuleSummary({ summary }: { summary: RulePage["default_summary"] | undefined }) {
  if (!summary) return null;
  if (summary.total === 0) return <p className="text-sm text-muted-foreground">No default rules registered.</p>;
  const counts = summary.evaluation_status_counts ?? {
    pending: 0,
    complete: 0,
    insufficient_coverage: 0,
    insufficient_history: 0,
  };
  const waiting = [
    counts.pending > 0 ? `${counts.pending} awaiting their first check` : "",
    counts.insufficient_history > 0 ? `${counts.insufficient_history} waiting for more history` : "",
    counts.insufficient_coverage > 0 ? `${counts.insufficient_coverage} waiting for complete usage data` : "",
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <div className="space-y-1 py-1">
      <p className="text-sm">
        {summary.enabled} of {summary.total} default rules enabled
      </p>
      <p className="text-xs text-muted-foreground">
        Defaults cover daily changes across keys, teams, users, models, and the gateway.
      </p>
      {waiting && <p className="text-xs text-muted-foreground">{waiting}</p>}
    </div>
  );
}

const EVALUATION_LABELS = {
  pending: "Awaiting first evaluation",
  complete: "Evaluated",
  insufficient_coverage: "Waiting for complete usage data",
  insufficient_history: "Waiting for more history",
};

function RuleRow({
  rule,
  confirming,
  pending,
  onEdit,
  onDelete,
  onCancel,
}: {
  rule: UsageRule;
  confirming: boolean;
  pending: boolean;
  onEdit: () => void;
  onDelete: () => void;
  onCancel: () => void;
}) {
  const scopeId = rule.scope_id ?? "";
  const displayId = scopeId.length > 32 ? `${scopeId.slice(0, 12)}…${scopeId.slice(-6)}` : scopeId;
  return (
    <li className="flex items-center justify-between gap-3 py-3">
      <div className="min-w-0">
        <p className="font-medium truncate">
          {rule.name}
          {!rule.enabled && <span className="ml-2 font-normal text-muted-foreground">Disabled</span>}
        </p>
        <p className="text-xs text-muted-foreground truncate">
          {SCOPE_LABELS[rule.scope_type]}
          {displayId ? ` · ${displayId}` : ""} · {METRIC_LABELS[rule.metric]} · {rule.multiplier}× {rule.direction}
        </p>
        {rule.enabled && (
          <p className="mt-1 text-xs text-muted-foreground">
            {EVALUATION_LABELS[rule.evaluation_status ?? "pending"]}
            {rule.last_evaluated_day ? ` · ${rule.last_evaluated_day} UTC` : ""}
          </p>
        )}
      </div>
      <div className="flex shrink-0 gap-1">
        <Button size="sm" variant="ghost" onClick={confirming ? onCancel : onEdit}>
          {confirming ? "Cancel" : "Edit"}
        </Button>
        {!rule.is_default && (
          <Button size="sm" variant={confirming ? "destructive" : "ghost"} disabled={pending} onClick={onDelete}>
            {confirming ? "Confirm delete" : "Delete"}
          </Button>
        )}
      </div>
    </li>
  );
}
