import { useTranslation } from "react-i18next";
import { useQueryState } from "nuqs";
import { Badge } from "@/components/ui/badge";
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table";
import { Decision, useExternalQuery } from "./useExternalAgents";
import { displayTime, pageParser } from "./forms";
import { Pagination } from "./BotsPanel";

export function DecisionsPanel({ token, sourceId }: { token: string; sourceId: string }) {
  const { t } = useTranslation();

  const [page, setPage] = useQueryState("decision_page", pageParser);
  const current = page;
  const query = useExternalQuery<Decision[]>(token, "/decisions", {
    source_id: sourceId,
    limit: 20,
    offset: (current - 1) * 20,
  });
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold">{t("agents.external.governanceDecisions")}</h2>
        <p className="text-sm text-muted-foreground">
          {t("agents.external.contentfreeAuditLogShadowModeAllowsRequestsWhileRecordingWhatWouldBeDenied")}
        </p>
      </div>
      {query.error && (
        <p role="alert" className="text-destructive">
          {query.error.message}
        </p>
      )}
      <div className="rounded-xl border bg-card overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              {[
                t("agents.external.time"),
                t("agents.external.bot"),
                t("agents.external.member"),
                t("agents.external.frameType"),
                t("agents.external.tools"),
                t("agents.external.verdict"),
                t("agents.external.reason"),
                t("agents.external.enforced"),
              ].map((name) => (
                <TableHead key={name}>{name}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {
              <>
                {query.isPending && (
                  <TableRow>
                    <TableCell colSpan={8}>{t("agents.external.loadingDecisions")}</TableCell>
                  </TableRow>
                )}
                {!query.isPending && (
                  <>
                    {!query.data?.length && (
                      <TableRow>
                        <TableCell colSpan={8} className="py-12 text-center text-muted-foreground">
                          {t("agents.external.noDecisionsYet")}
                        </TableCell>
                      </TableRow>
                    )}
                    {!!query.data?.length &&
                      query.data.map((decision) => (
                        <TableRow key={decision.request_id}>
                          <TableCell className="min-w-36">{displayTime(decision.created_at)}</TableCell>
                          <TableCell>{decision.external_agent_id}</TableCell>
                          <TableCell>{decision.actor_id || t("agents.external.unknown")}</TableCell>
                          <TableCell>{decision.frame_type}</TableCell>
                          <TableCell>{decision.tool_names.join(", ") || t("agents.external.none")}</TableCell>
                          <TableCell>
                            <div className="space-y-1">
                              <Badge variant={decision.verdict === "deny" ? "destructive" : "outline"}>
                                {decision.verdict}
                              </Badge>
                              {!decision.enforced && decision.would_verdict === "deny" && (
                                <Badge variant="destructive">{t("agents.external.wouldDenyShadow")}</Badge>
                              )}
                            </div>
                          </TableCell>
                          <TableCell className="min-w-56">
                            {decision.reason || t("agents.external.policyAllowsThisRequest")}
                          </TableCell>
                          <TableCell>
                            {decision.enforced ? t("agents.external.yes") : t("agents.external.no")}
                          </TableCell>
                        </TableRow>
                      ))}
                  </>
                )}
              </>
            }
          </TableBody>
        </Table>
      </div>
      <Pagination
        page={current}
        count={query.data?.length ?? 0}
        onChange={(next) => {
          void setPage(next);
        }}
      />
    </div>
  );
}
