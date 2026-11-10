import { useMutation, useQuery, useQueryClient, type QueryKey } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { getSkillScan, startSkillScan } from "@/components/networking";
import type { components } from "@/lib/http/schema";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface SkillScanPanelProps {
  id: string;
  accessToken: string;
  queryScope: QueryKey;
  pending: boolean;
}

export default function SkillScanPanel({ id, accessToken, queryScope, pending }: SkillScanPanelProps) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const queryKey = [...queryScope, "submission", id, "scan"];
  const result = useQuery({
    queryKey,
    queryFn: () => getSkillScan(accessToken, id),
    refetchInterval: (query) => (query.state.data?.scan?.status === "running" ? 2000 : false),
  });
  const start = useMutation({
    mutationFn: () => startSkillScan(accessToken, id),
    onSettled: () => queryClient.invalidateQueries({ queryKey, exact: true }),
  });
  const scan = result.data?.scan;
  const report = scan?.report;
  const running = start.isPending || scan?.status === "running";
  return (
    <section aria-label={t("skills.scanning.title")} className="space-y-3 rounded-lg border p-4">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold">{t("skills.scanning.title")}</h3>
        {pending && (
          <Button
            variant="outline"
            disabled={running || result.isError || !result.data?.configured}
            onClick={() => start.mutate()}
          >
            {t(running ? "skills.scanning.running" : "skills.scanning.scan")}
          </Button>
        )}
      </div>
      <p className="text-xs text-muted-foreground">{t("skills.scanning.advisory")}</p>
      {result.isPending && <p className="text-sm">{t("skills.scanning.loading")}</p>}
      {result.data && !result.data.configured && (
        <p className="text-sm text-muted-foreground">{t("skills.scanning.unconfigured")}</p>
      )}
      {result.isError && (
        <div role="alert" className="text-sm text-destructive">
          {t("skills.scanning.loadFailed")}
          <Button variant="link" onClick={() => void result.refetch()}>
            {t("skills.submissions.retry")}
          </Button>
        </div>
      )}
      {start.isError && (
        <p role="alert" className="text-sm text-destructive">
          {start.error.message}
        </p>
      )}
      {scan && (
        <>
          <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
            <Badge variant="outline">{t(`skills.scanning.status.${scan.status}`)}</Badge>
            <span>{t("skills.scanning.requestedBy", { name: scan.requested_by })}</span>
            <span>{new Date(scan.started_at).toLocaleString()}</span>
          </div>
          <p className="break-all text-xs text-muted-foreground">
            {scan.archive_sha256
              ? t("skills.scanning.archiveDigest", { digest: scan.archive_sha256 })
              : t("skills.scanning.mutableSource")}
          </p>
          {scan.error && (
            <p role="alert" className="text-sm text-destructive">
              {scan.error}
            </p>
          )}
        </>
      )}
      {report && <ScanReport report={report} />}
    </section>
  );
}

function ScanReport({ report }: { report: components["schemas"]["SkillScanReport"] }) {
  const { t } = useTranslation();
  return (
    <>
      <p className="text-sm">
        {t("skills.scanning.summary", {
          count: report.findings.length,
          score: report.risk_score,
          version: report.scanner_version,
        })}
      </p>
      <p className="text-xs text-muted-foreground">
        {t(report.scan_mode === "static+llm" ? "skills.scanning.semantic" : "skills.scanning.static")}
      </p>
      {!report.complete && (
        <p role="alert" className="text-sm text-warning">
          {t("skills.scanning.incomplete")}
        </p>
      )}
      {(report.limitations ?? []).map((limitation, index) => (
        <p key={index} className="text-xs text-muted-foreground">
          {limitation}
        </p>
      ))}
      {report.findings.length === 0 && <p className="text-sm">{t("skills.scanning.noFindings")}</p>}
      <div className="max-h-80 space-y-3 overflow-y-auto">
        {report.findings.map((finding, index) => (
          <article key={index} className="space-y-1 rounded border p-3 text-sm">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline">{finding.severity}</Badge>
              <span className="font-mono text-xs">{finding.rule_id}</span>
            </div>
            <p className="whitespace-pre-wrap break-words font-medium">{finding.message}</p>
            <p className="break-all font-mono text-xs text-muted-foreground">
              {finding.file}
              {finding.start_line ? `:${finding.start_line}` : ""}
            </p>
            {finding.explanation && <p className="whitespace-pre-wrap break-words">{finding.explanation}</p>}
            {finding.remediation && <p className="whitespace-pre-wrap break-words">{finding.remediation}</p>}
          </article>
        ))}
      </div>
    </>
  );
}
