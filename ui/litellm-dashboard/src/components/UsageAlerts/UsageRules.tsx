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
import { METRIC_LABELS, RuleInput, SCOPE_LABELS, UsageRule } from "./usageAlertApi";
import UsageRuleForm from "./UsageRuleForm";

export default function UsageRules({ accessToken }: { accessToken: string | null }) {
  const { data: user } = useCurrentUser(accessToken);
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState<UsageRule | null | undefined>(undefined);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: ["usageAlertRules", getProxyBaseUrl(), user?.user_id, user?.user_role, offset],
    enabled: open && Boolean(accessToken && user?.user_id),
    queryFn: async ({ signal }) => {
      const { data } = await backgroundFetchClient.GET("/usage/alert-rules", {
        params: { query: { limit: 20, offset } },
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
  let title = "Alert rules";
  if (editing !== undefined) title = editing ? "Edit rule" : "New rule";
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
          <DialogTitle>{title}</DialogTitle>
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
            <div className="flex justify-end">
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
              <p className="text-muted-foreground">No alert rules.</p>
            )}
            {query.isSuccess && (
              <ul className="divide-y">
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
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">{total} rules</span>
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
          </>
        )}
      </DialogContent>
    </Dialog>
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
  return (
    <li className="flex items-center justify-between gap-3 py-3">
      <div className="min-w-0">
        <p className="font-medium truncate">
          {rule.name}
          {!rule.enabled && <span className="ml-2 font-normal text-muted-foreground">Disabled</span>}
        </p>
        <p className="text-xs text-muted-foreground truncate">
          {SCOPE_LABELS[rule.scope_type]}
          {rule.scope_id ? ` · ${rule.scope_id}` : ""} · {METRIC_LABELS[rule.metric]} · {rule.multiplier}×{" "}
          {rule.direction}
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
        <Button size="sm" variant={confirming ? "destructive" : "ghost"} disabled={pending} onClick={onDelete}>
          {confirming ? "Confirm delete" : "Delete"}
        </Button>
      </div>
    </li>
  );
}
