import { useState } from "react";
import { useQuery, type QueryKey } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { downloadBlob } from "@/components/EntityUsageExport/utils";
import {
  downloadSkillSubmission,
  getSkillSubmission,
  listSkillSubmissions,
  reviewSkillSubmission,
} from "@/components/networking";
import type { SkillHubSkill, SkillSubmission } from "@/components/claude_code_plugins/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { useUrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable/useUrlTableState";
import SkillScanPanel from "./SkillScanPanel";
import { toast } from "@/lib/toast";

const METADATA_FIELDS = ["name", "version", "description", "source", "keywords", "domain", "namespace"] as const;

const displayValue = (value: unknown): string => {
  if (value === null || value === undefined || value === "") return "";
  return typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
};

const STATUS_CLASSES = {
  pending_review: "border-warning/30 text-warning",
  approved: "border-success/30 text-success",
  rejected: "border-destructive/30 text-destructive",
};

const SUBMISSION_TABLE_STATE: UrlTableStateOptions<never> = {
  keyPrefix: "submissions_",
  defaultPageSize: 20,
  maxPageSize: 100,
  sortFields: ["created_at"],
  defaultSort: { id: "created_at", desc: true },
  filterColumns: [],
};

function SubmissionStatus({ status }: { status: SkillSubmission["status"] }) {
  const { t } = useTranslation();
  return (
    <Badge variant="outline" className={STATUS_CLASSES[status]}>
      {t(`skills.submissions.status.${status}`)}
    </Badge>
  );
}

function SubmissionSummary({ submission }: { submission: SkillSubmission }) {
  const { t } = useTranslation();
  return (
    <>
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <SubmissionStatus status={submission.status} />
        <span>{t(`skills.submissions.operation.${submission.operation}`)}</span>
        <span className="text-muted-foreground">
          {t("skills.submissions.submittedBy", { name: submission.submitted_by })}
        </span>
      </div>
      {submission.status === "pending_review" && (
        <p className="text-sm text-muted-foreground">{t("skills.submissions.pendingNotice")}</p>
      )}
      {submission.rejection_reason && (
        <p className="text-sm text-destructive">
          {t("skills.submissions.rejectionReason", { reason: submission.rejection_reason })}
        </p>
      )}
      {submission.reviewed_at && (
        <p className="text-sm text-muted-foreground">
          {t("skills.submissions.reviewedBy", {
            name: submission.reviewed_by,
            date: new Date(submission.reviewed_at).toLocaleString(),
          })}
        </p>
      )}
    </>
  );
}

function SubmissionComparison({
  submission,
  current,
}: {
  submission: SkillSubmission;
  current?: SkillHubSkill | null;
}) {
  const { t } = useTranslation();
  return (
    <div className="overflow-x-auto">
      <table className="w-full table-fixed border-collapse text-sm">
        <thead>
          <tr className="border-b">
            <th className="w-1/5 p-2 text-left">{t("skills.submissions.field")}</th>
            <th className="p-2 text-left">{t("skills.submissions.current")}</th>
            <th className="p-2 text-left">{t("skills.submissions.proposed")}</th>
          </tr>
        </thead>
        <tbody>
          {METADATA_FIELDS.map((field) => (
            <tr key={field} className="border-b align-top">
              <th className="p-2 text-left font-medium">{t(`skills.submissions.fields.${field}`)}</th>
              <td className="p-2">
                <pre className="whitespace-pre-wrap break-words font-mono text-xs">
                  {displayValue(current?.[field]) || t("skills.submissions.notSet")}
                </pre>
              </td>
              <td className="p-2">
                <pre className="whitespace-pre-wrap break-words font-mono text-xs">
                  {field === "version" && submission.operation === "update" && !submission.proposal.version
                    ? t("skills.submissions.nextPatchVersion")
                    : displayValue(submission.proposal[field]) || t("skills.submissions.notSet")}
                </pre>
              </td>
            </tr>
          ))}
          <tr className="border-b align-top">
            <th className="p-2 text-left font-medium">{t("skills.submissions.fields.agent_skill")}</th>
            <td className="p-2 text-muted-foreground">{t("skills.submissions.descriptorUnavailable")}</td>
            <td className="p-2">
              <pre className="whitespace-pre-wrap break-words font-mono text-xs">
                {displayValue(submission.proposal.agent_skill) || t("skills.submissions.notSet")}
              </pre>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

interface SubmissionReviewProps {
  id: string;
  accessToken: string;
  queryScope: QueryKey;
  canReview: boolean;
  onClose: () => void;
  onChanged: () => Promise<void>;
}

function SubmissionReview({ id, accessToken, queryScope, canReview, onClose, onChanged }: SubmissionReviewProps) {
  const { t } = useTranslation();
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const detail = useQuery({
    queryKey: [...queryScope, "submission", id],
    queryFn: () => getSkillSubmission(accessToken, id),
  });
  const { submission, current_skill: current } = detail.data ?? {};
  const canDecide = canReview && submission?.status === "pending_review";
  const review = async (action: "approve" | "reject") => {
    if (!canDecide || busy) return;
    setBusy(true);
    setError(null);
    try {
      await reviewSkillSubmission(
        accessToken,
        id,
        action,
        action === "reject" ? { reason: reason.trim() || null } : undefined,
      );
      onClose();
      await onChanged();
      toast.success(t(action === "approve" ? "skills.submissions.approved" : "skills.submissions.rejected"));
    } catch (err) {
      setError(err instanceof Error ? err.message : t("skills.submissions.actionFailed"));
      await onChanged();
    } finally {
      setBusy(false);
    }
  };
  const download = async () => {
    if (!submission || downloading) return;
    setDownloading(true);
    setError(null);
    try {
      downloadBlob(await downloadSkillSubmission(accessToken, id), `${submission.skill_name}.zip`);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("skills.submissions.downloadFailed"));
    } finally {
      setDownloading(false);
    }
  };
  return (
    <Dialog open onOpenChange={(open) => !open && !busy && onClose()}>
      <DialogContent className="top-8 max-h-[calc(100dvh-4rem)] translate-y-0 overflow-y-auto sm:max-w-[960px]">
        <DialogHeader>
          <DialogTitle>{t("skills.submissions.reviewTitle", { name: submission?.skill_name ?? "" })}</DialogTitle>
        </DialogHeader>
        {detail.isPending && <p>{t("skills.submissions.loading")}</p>}
        {detail.isError && (
          <div role="alert" className="text-destructive">
            {t("skills.submissions.loadFailed")}
            <Button variant="link" onClick={() => void detail.refetch()}>
              {t("skills.submissions.retry")}
            </Button>
          </div>
        )}
        {submission && (
          <>
            <SubmissionSummary submission={submission} />
            {!current && <p className="text-sm text-muted-foreground">{t("skills.submissions.noCurrent")}</p>}
            <SubmissionComparison submission={submission} current={current} />
            {submission.has_archive && submission.proposal.agent_skill && (
              <Button variant="outline" onClick={download} disabled={downloading}>
                {t(downloading ? "skills.submissions.downloading" : "skills.submissions.download")}
              </Button>
            )}
            {canReview && (
              <SkillScanPanel
                id={id}
                accessToken={accessToken}
                queryScope={queryScope}
                pending={submission.status === "pending_review"}
              />
            )}
            {canDecide && (
              <>
                <label htmlFor="skill-rejection-reason" className="text-sm font-medium">
                  {t("skills.submissions.reasonLabel")}
                </label>
                <Textarea
                  id="skill-rejection-reason"
                  value={reason}
                  maxLength={2000}
                  onChange={(event) => setReason(event.target.value)}
                  disabled={busy}
                />
                <div className="flex justify-end gap-2">
                  <Button
                    variant="destructive"
                    onClick={() => void review("reject")}
                    disabled={busy || detail.isFetching || detail.isError}
                  >
                    {t("skills.submissions.reject")}
                  </Button>
                  <Button onClick={() => void review("approve")} disabled={busy || detail.isFetching || detail.isError}>
                    {t("skills.submissions.approve")}
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

interface SkillSubmissionsProps {
  accessToken: string;
  queryScope: QueryKey;
  canReview: boolean;
  adminView: boolean;
  onChanged: () => Promise<void>;
}

export default function SkillSubmissions({
  accessToken,
  queryScope,
  canReview,
  adminView,
  onChanged,
}: SkillSubmissionsProps) {
  const { t } = useTranslation();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const { pagination, onPaginationChange } = useUrlTableState(SUBMISSION_TABLE_STATE);
  const { pageIndex, pageSize } = pagination;
  const submissions = useQuery({
    queryKey: [...queryScope, "submissions", pageIndex, pageSize],
    queryFn: () => listSkillSubmissions(accessToken, { limit: pageSize, offset: pageIndex * pageSize }),
  });
  const isEmpty = submissions.isSuccess && submissions.data.submissions.length === 0;
  const changePage = (page: number) => {
    setSelectedId(null);
    onPaginationChange({ pageIndex: page, pageSize });
  };
  return (
    <div className="space-y-4 py-4">
      <p className="text-sm text-muted-foreground">
        {t(adminView ? "skills.submissions.adminIntro" : "skills.submissions.authorIntro")}
      </p>
      {submissions.isPending && <p>{t("skills.submissions.loading")}</p>}
      {submissions.isError && (
        <div role="alert" className="text-sm text-destructive">
          {t("skills.submissions.loadFailed")}
          <Button variant="link" onClick={() => void submissions.refetch()}>
            {t("skills.submissions.retry")}
          </Button>
        </div>
      )}
      {isEmpty && (
        <p className="py-8 text-center text-sm text-muted-foreground">
          {t(pageIndex === 0 ? "skills.submissions.empty" : "skills.submissions.emptyPage")}
        </p>
      )}
      {submissions.data?.submissions.map((submission) => (
        <div key={submission.id} className="rounded-lg border bg-card p-4">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0 space-y-2">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="break-all text-sm font-semibold">{submission.skill_name}</h2>
                <SubmissionStatus status={submission.status} />
                <span className="text-xs text-muted-foreground">
                  {t(`skills.submissions.operation.${submission.operation}`)}
                </span>
              </div>
              <p className="text-sm text-muted-foreground">{submission.proposal.description}</p>
              <p className="text-xs text-muted-foreground">
                {t("skills.submissions.submittedBy", { name: submission.submitted_by })} ·{" "}
                {new Date(submission.created_at).toLocaleString()}
              </p>
              {submission.rejection_reason && (
                <p className="text-sm text-destructive">
                  {t("skills.submissions.rejectionReason", { reason: submission.rejection_reason })}
                </p>
              )}
            </div>
            <Button variant="outline" size="sm" onClick={() => setSelectedId(submission.id)}>
              {t(
                canReview && submission.status === "pending_review"
                  ? "skills.submissions.review"
                  : "skills.submissions.details",
              )}
            </Button>
          </div>
        </div>
      ))}
      <div className="flex items-center justify-end gap-3">
        <span className="text-sm text-muted-foreground">{t("skills.submissions.page", { page: pageIndex + 1 })}</span>
        <Button
          variant="outline"
          disabled={pageIndex === 0 || submissions.isFetching}
          onClick={() => changePage(pageIndex - 1)}
        >
          {t("skills.submissions.previous")}
        </Button>
        <Button
          variant="outline"
          disabled={!submissions.data?.has_more || submissions.isFetching || submissions.isError}
          onClick={() => changePage(pageIndex + 1)}
        >
          {t("skills.submissions.next")}
        </Button>
      </div>
      {selectedId && (
        <SubmissionReview
          key={selectedId}
          id={selectedId}
          accessToken={accessToken}
          queryScope={queryScope}
          canReview={canReview}
          onClose={() => setSelectedId(null)}
          onChanged={onChanged}
        />
      )}
    </div>
  );
}
