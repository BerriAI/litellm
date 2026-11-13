import { useState } from "react";
import { useQuery, useQueryClient, type QueryKey } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { Wallet } from "lucide-react";
import { z } from "zod";
import { apiClient } from "@/components/networking";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderControls, PageHeaderDescription, PageHeaderTitle } from "@/components/shared/PageHeader";
import { useUrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable/useUrlTableState";
import { FormField } from "@/components/shared/form/FormField";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useZodForm } from "@/lib/forms/useZodForm";
import type { components } from "@/lib/http/schema";
import { toast } from "@/lib/toast";
import { isProxyAdminRole, isProxyAdminTierRole } from "@/utils/roles";

type Increase = components["schemas"]["BudgetIncreaseRequest"];
type Assignment = components["schemas"]["BudgetAssignment"];
const PATH = "/management/v1/budget_increase_requests";
const TABLE: UrlTableStateOptions<never> = {
  keyPrefix: "budget_requests_",
  defaultPageSize: 20,
  maxPageSize: 100,
  sortFields: ["created_at"],
  defaultSort: { id: "created_at", desc: true },
  filterColumns: [],
};
const FORM_FIELDS = {
  amount: z.number().positive().finite(),
  duration: z.enum(["1mo", "12mo"]),
  expiry: z.string().min(1),
  reason: z.string().trim().min(1).max(2000),
};
const FORM = z.object(FORM_FIELDS);
const dollars = (value: number) =>
  new Intl.NumberFormat(undefined, { style: "currency", currency: "USD" }).format(value);
const date = (value: string) => new Date(value).toLocaleString();

type RequestFormProps = {
  accessToken: string;
  scope: QueryKey;
  onChanged: () => Promise<void>;
  onClose: () => void;
};

function RequestForm({ scope, ...props }: RequestFormProps) {
  const { t } = useTranslation();
  const assignment = useQuery({
    queryKey: [...scope, "me"],
    queryFn: ({ signal }) => apiClient.get<Assignment>(`${PATH}/me`, { accessToken: props.accessToken, signal }),
  });
  if (assignment.data) return <ReadyRequestForm {...props} assignment={assignment.data} />;
  return (
    <Dialog open onOpenChange={(open) => !open && props.onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{t("budgetRequests.request")}</DialogTitle>
        </DialogHeader>
        {assignment.error ? <p role="alert">{assignment.error.message}</p> : <p>{t("budgetRequests.loading")}</p>}
      </DialogContent>
    </Dialog>
  );
}

function ReadyRequestForm({
  accessToken,
  assignment,
  onChanged,
  onClose,
}: Omit<RequestFormProps, "scope"> & { assignment: Assignment }) {
  const { t } = useTranslation();
  const [error, setError] = useState<string | null>(null);
  const form = useZodForm(FORM, {
    defaultValues: { amount: 50, duration: assignment.windows[0]?.budget_duration ?? "1mo", expiry: "", reason: "" },
  });
  const allowed = assignment.effective_budget.cap_policy === "pause_allow_increase";
  const windows = assignment.windows;
  const save = form.handleSubmit(async (values) => {
    setError(null);
    try {
      await apiClient.post<Increase>(PATH, {
        accessToken,
        body: {
          amount: values.amount,
          duration: values.duration,
          expires_at: new Date(`${values.expiry}Z`).toISOString(),
          reason: values.reason,
        } satisfies components["schemas"]["BudgetIncreaseInput"],
      });
      await onChanged();
      onClose();
      toast.success(t("budgetRequests.submitted"));
    } catch (err) {
      setError(err instanceof Error ? err.message : t("budgetRequests.failed"));
    }
  });
  return (
    <Dialog open onOpenChange={(open) => !open && !form.formState.isSubmitting && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{t("budgetRequests.request")}</DialogTitle>
        </DialogHeader>
        <>
          <p className="text-sm text-muted-foreground">{t("budgetRequests.requestNotice")}</p>
          {windows.map((window) => (
            <p key={window.budget_duration} className="text-sm">
              {t(window.budget_duration === "1mo" ? "budgetProfiles.monthlyLimit" : "budgetProfiles.annualLimit")}:{" "}
              {dollars(window.max_budget)}
            </p>
          ))}
          {!allowed ? (
            <p>{t("budgetRequests.disabledPolicy")}</p>
          ) : (
            <form onSubmit={save} className="space-y-4">
              <FormField control={form.control} name="amount" label={t("budgetRequests.amount")}>
                {({ value, onChange, ...field }) => (
                  <Input
                    {...field}
                    type="number"
                    min={0.01}
                    step={0.01}
                    value={value}
                    onChange={(event) => onChange(event.target.valueAsNumber)}
                  />
                )}
              </FormField>
              <FormField control={form.control} name="duration" label={t("budgetRequests.duration")}>
                {(field) => (
                  <Select
                    value={field.value}
                    onValueChange={field.onChange}
                    items={windows.map((window) => ({
                      value: window.budget_duration,
                      label: t(
                        window.budget_duration === "1mo" ? "budgetProfiles.monthlyLimit" : "budgetProfiles.annualLimit",
                      ),
                    }))}
                  >
                    <SelectTrigger id={field.id}>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {windows.map((window) => (
                        <SelectItem key={window.budget_duration} value={window.budget_duration}>
                          {t(
                            window.budget_duration === "1mo"
                              ? "budgetProfiles.monthlyLimit"
                              : "budgetProfiles.annualLimit",
                          )}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </FormField>
              <FormField control={form.control} name="expiry" label={t("budgetRequests.expiryUtc")}>
                {(field) => <Input {...field} type="datetime-local" />}
              </FormField>
              <FormField control={form.control} name="reason" label={t("budgetRequests.reason")}>
                {(field) => <Textarea {...field} maxLength={2000} />}
              </FormField>
              <Button type="submit" disabled={form.formState.isSubmitting || windows.length === 0}>
                {t("budgetRequests.submit")}
              </Button>
            </form>
          )}
        </>
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}

function RequestFacts({ row }: { row: Increase }) {
  const { t } = useTranslation();
  return (
    <>
      <Badge variant="outline">{t(`budgetRequests.status.${row.status}`)}</Badge>
      <dl className="grid grid-cols-2 gap-3 text-sm">
        <dt>{t("budgetRequests.requester")}</dt>
        <dd className="break-all">{row.submitted_by}</dd>
        <dt>{t("budgetRequests.amount")}</dt>
        <dd>{dollars(row.amount)}</dd>
        <dt>{t("budgetRequests.duration")}</dt>
        <dd>{t(row.duration === "1mo" ? "budgetProfiles.monthlyLimit" : "budgetProfiles.annualLimit")}</dd>
        <dt>{t("budgetRequests.expiry")}</dt>
        <dd>{date(row.expires_at)}</dd>
      </dl>
      <p className="whitespace-pre-wrap text-sm">{row.reason}</p>
      {row.rejection_reason && <p className="text-destructive">{row.rejection_reason}</p>}
      {row.reviewed_at && (
        <p className="text-sm text-muted-foreground">
          {t("budgetRequests.reviewedBy", { name: row.reviewed_by, date: date(row.reviewed_at) })}
        </p>
      )}
    </>
  );
}

function Review({
  id,
  accessToken,
  scope,
  canReview,
  userId,
  onClose,
  onChanged,
}: {
  id: string;
  accessToken: string;
  scope: QueryKey;
  canReview: boolean;
  userId: string;
  onClose: () => void;
  onChanged: () => Promise<void>;
}) {
  const { t } = useTranslation();
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const detail = useQuery({
    queryKey: [...scope, id],
    queryFn: ({ signal }) => apiClient.get<Increase>(`${PATH}/${encodeURIComponent(id)}`, { accessToken, signal }),
  });
  const row = detail.data;
  const canDecide = canReview && row?.status === "pending_review" && row.submitted_by !== userId;
  const current = useQuery({
    queryKey: [...scope, "assignment", row?.submitted_by],
    enabled: canReview && Boolean(row),
    queryFn: ({ signal }) =>
      apiClient.get<Assignment>(`/management/v1/budget_assignments/user/${encodeURIComponent(row!.submitted_by)}`, {
        accessToken,
        signal,
      }),
  });
  const detailsUnavailable = detail.isFetching || current.isFetching || current.isError;
  const approvalDisabled = busy || detailsUnavailable;
  const decide = async (action: "approve" | "reject") => {
    if (!canDecide || busy) return;
    setBusy(true);
    setError(null);
    try {
      await apiClient.put<Increase>(`${PATH}/${encodeURIComponent(id)}/${action}`, {
        accessToken,
        ...(action === "reject" ? { body: { reason: reason.trim() } } : {}),
      });
      await onChanged();
      onClose();
      toast.success(t(action === "approve" ? "budgetRequests.approved" : "budgetRequests.rejected"));
    } catch (err) {
      setError(err instanceof Error ? err.message : t("budgetRequests.failed"));
      await onChanged();
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog open onOpenChange={(open) => !open && !busy && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{t("budgetRequests.details")}</DialogTitle>
        </DialogHeader>
        {detail.isPending && <p>{t("budgetRequests.loading")}</p>}
        {detail.error && <p role="alert">{detail.error.message}</p>}
        {row && (
          <>
            <RequestFacts row={row} />
            {current.data && (
              <div className="rounded border p-3 text-sm">
                <p className="font-medium">{t("budgetRequests.currentLimits")}</p>
                {current.data.windows.map((window) => (
                  <p key={window.budget_duration}>
                    {t(window.budget_duration === "1mo" ? "budgetProfiles.monthlyLimit" : "budgetProfiles.annualLimit")}
                    : {dollars(window.max_budget)}
                  </p>
                ))}
              </div>
            )}
            {current.error && <p role="alert">{current.error.message}</p>}
            {canReview && row.submitted_by === userId && <p>{t("budgetRequests.noSelfReview")}</p>}
            {canDecide && (
              <>
                <label htmlFor="budget-rejection-reason" className="text-sm font-medium">
                  {t("budgetRequests.rejectionReason")}
                </label>
                <Textarea
                  id="budget-rejection-reason"
                  value={reason}
                  maxLength={2000}
                  onChange={(event) => setReason(event.target.value)}
                  disabled={busy}
                />
                <div className="flex justify-end gap-2">
                  <Button
                    variant="destructive"
                    onClick={() => void decide("reject")}
                    disabled={busy || !reason.trim() || detail.isFetching}
                  >
                    {t("budgetRequests.reject")}
                  </Button>
                  <Button onClick={() => void decide("approve")} disabled={approvalDisabled}>
                    {t("budgetRequests.approve")}
                  </Button>
                </div>
              </>
            )}
          </>
        )}
        {error && (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        )}
      </DialogContent>
    </Dialog>
  );
}

export default function BudgetRequests({
  accessToken,
  userRole,
  userId,
  isViewOnly,
}: {
  accessToken: string;
  userRole: string;
  userId: string;
  isViewOnly: boolean;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [creating, setCreating] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const canReview = !isViewOnly && isProxyAdminRole(userRole);
  const scope = ["budget-requests", accessToken, userRole, userId, isViewOnly];
  const { pagination, onPaginationChange } = useUrlTableState(TABLE);
  const { pageIndex, pageSize } = pagination;
  const list = useQuery({
    queryKey: [...scope, "list", pageIndex, pageSize],
    queryFn: ({ signal }) =>
      apiClient.get<components["schemas"]["BudgetIncreaseList"]>(PATH, {
        accessToken,
        signal,
        query: { limit: pageSize, offset: pageIndex * pageSize },
      }),
  });
  const changed = async () => {
    await queryClient.invalidateQueries({ queryKey: scope });
    await queryClient.invalidateQueries({ queryKey: ["budget-assignment"] });
  };
  return (
    <Page className="h-full">
      <PageHeader>
        <PageHeaderTitle>
          <Wallet />
          {t("budgetRequests.title")}
        </PageHeaderTitle>
        <PageHeaderDescription>
          {t(isProxyAdminTierRole(userRole) ? "budgetRequests.adminIntro" : "budgetRequests.userIntro")}
        </PageHeaderDescription>
        <PageHeaderControls>
          <Button variant="outline" onClick={() => void changed()} disabled={list.isFetching}>
            {t("budgetRequests.refresh")}
          </Button>
          {!isViewOnly && <Button onClick={() => setCreating(true)}>{t("budgetRequests.request")}</Button>}
        </PageHeaderControls>
      </PageHeader>
      <div className="space-y-4 overflow-y-auto py-4">
        {list.isPending && <p>{t("budgetRequests.loading")}</p>}
        {list.error && <p role="alert">{list.error.message}</p>}
        {list.data?.requests.length === 0 && (
          <p className="py-8 text-center text-muted-foreground">{t("budgetRequests.empty")}</p>
        )}
        {list.data?.requests.map((row) => (
          <div key={row.id} className="flex items-start justify-between gap-4 rounded-lg border bg-card p-4">
            <div className="min-w-0 space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="break-all text-sm font-semibold">{row.submitted_by}</h2>
                <Badge variant="outline">{t(`budgetRequests.status.${row.status}`)}</Badge>
              </div>
              <p className="text-sm">
                {dollars(row.amount)} ·{" "}
                {t(row.duration === "1mo" ? "budgetProfiles.monthlyLimit" : "budgetProfiles.annualLimit")}
              </p>
              <p className="text-sm text-muted-foreground">{row.reason}</p>
              <p className="text-xs text-muted-foreground">
                {t("budgetRequests.expires", { date: date(row.expires_at) })}
              </p>
            </div>
            <Button variant="outline" size="sm" onClick={() => setSelected(row.id)}>
              {t(canReview && row.status === "pending_review" ? "budgetRequests.review" : "budgetRequests.details")}
            </Button>
          </div>
        ))}
        <div className="flex items-center justify-end gap-3">
          <span className="text-sm">{t("skills.submissions.page", { page: pageIndex + 1 })}</span>
          <Button
            variant="outline"
            disabled={pageIndex === 0 || list.isFetching}
            onClick={() => onPaginationChange({ pageIndex: pageIndex - 1, pageSize })}
          >
            {t("skills.submissions.previous")}
          </Button>
          <Button
            variant="outline"
            disabled={!list.data?.has_more || list.isFetching || list.isError}
            onClick={() => onPaginationChange({ pageIndex: pageIndex + 1, pageSize })}
          >
            {t("skills.submissions.next")}
          </Button>
        </div>
      </div>
      {creating && (
        <RequestForm accessToken={accessToken} scope={scope} onChanged={changed} onClose={() => setCreating(false)} />
      )}
      {selected && (
        <Review
          id={selected}
          accessToken={accessToken}
          scope={scope}
          canReview={canReview}
          userId={userId}
          onChanged={changed}
          onClose={() => setSelected(null)}
        />
      )}
    </Page>
  );
}
