import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, X } from "lucide-react";
import { useCurrentUser } from "@/app/(dashboard)/hooks/currentUser/useCurrentUser";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { getProxyBaseUrl } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { hasCapability } from "@/utils/capabilities";
import { ApiError } from "@/lib/http/client";
import type { components } from "@/lib/http/schema";
import { Issue, IssueStatus, updateIssueStatus } from "./issueApi";

export default function IssueStatusActions({ issue, globalAccess = false }: { issue: Issue; globalAccess?: boolean }) {
  const { isViewOnly, accessToken, userRole } = useAuthorized();
  const { data: user } = useCurrentUser(accessToken);
  const client = useQueryClient();
  const mutation = useMutation({
    mutationFn: (status: IssueStatus) => updateIssueStatus(issue, status),
    onSuccess: (updated) => {
      const scope = ["issues", getProxyBaseUrl(), user?.user_id, user?.user_role];
      client.setQueryData<Issue>(
        [...scope, "detail", issue.id],
        (previous) =>
          previous && {
            ...updated,
            can_view_activity: updated.can_view_activity ?? previous.can_view_activity,
            can_update_status: updated.can_update_status ?? previous.can_update_status,
          },
      );
      client.setQueriesData<components["schemas"]["IssueList"]>(
        { queryKey: [...scope, "list"] },
        (page) =>
          page && {
            ...page,
            data: page.data.map((row) => (row.id === updated.id ? { ...row, status: updated.status } : row)),
          },
      );
      return Promise.all([
        client.invalidateQueries({ queryKey: ["issues"] }),
        client.invalidateQueries({ queryKey: ["deprecationAlerts"] }),
      ]);
    },
  });
  const inactiveRetirement = issue.kind === "model_deprecation" && issue.active === false;
  const permitted = issue.can_update_status ?? (globalAccess && hasCapability(userRole, "viewGlobalSpend"));
  if (isViewOnly || !permitted) return null;
  if (inactiveRetirement && issue.status !== "open") return null;
  const actions: IssueStatus[] = issue.status === "open" ? ["resolved", "dismissed"] : ["open"];
  const labels = { resolved: "Mark resolved", dismissed: "Dismiss", open: "Reopen" };
  const stale = mutation.error instanceof ApiError && mutation.error.status === 409;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        {actions.map((status) => (
          <Button
            key={status}
            variant={status === "dismissed" ? "ghost" : "outline"}
            size="sm"
            disabled={mutation.isPending}
            onClick={() => mutation.mutate(status)}
          >
            {status === "resolved" && <Check aria-hidden="true" className="size-3.5" />}
            {status === "dismissed" && <X aria-hidden="true" className="size-3.5" />}
            {labels[status]}
          </Button>
        ))}
      </div>
      {mutation.isError && (
        <p role="alert" className="max-w-sm text-xs text-destructive">
          {stale ? "This alert changed. Refresh it before trying again." : "Could not update this alert. Please retry."}
          {stale && (
            <Button
              variant="link"
              size="sm"
              onClick={() => {
                mutation.reset();
                void client.invalidateQueries({ queryKey: ["issues"] });
              }}
            >
              Refresh
            </Button>
          )}
        </p>
      )}
    </div>
  );
}
