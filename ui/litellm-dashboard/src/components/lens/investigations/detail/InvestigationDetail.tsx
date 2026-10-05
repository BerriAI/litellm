"use client";

import { Button } from "@/components/ui/button";

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { RunsTab } from "./RunsTab";
import { InvestigationSummary } from "./InvestigationSummary";
import { RunReport } from "./RunReport";
import type { OwnedFinding } from "../../model/inbox";
import { type Finding, type Lens } from "../../model/types";
import { FindingPanel } from "../FindingDetails";
import { useSectionRoute } from "../../route";
import { useRunSnapshot } from "../useRunSnapshot";

import { InvestigationActions, type InvestigationIntents } from "./InvestigationActions";
import { RunPicker } from "./RunPicker";
import { FindingsTab } from "./FindingsTab";
import { CriteriaTab } from "./CriteriaTab";
import { HistoryTab } from "./HistoryTab";

export type InvestigationDetailProps = InvestigationIntents & {
  readonly lens: Lens;
  readonly readOnly: boolean;
  readonly ready: boolean;
  readonly busy: boolean;
  readonly connected: boolean;
  readonly onReviewFinding: (owned: OwnedFinding, status: Finding["status"], reason: string) => void;
};

export function InvestigationDetail({
  lens,
  readOnly,
  ready,
  busy,
  connected,
  onCancelRun,
  onReviewFinding,
  ...intents
}: InvestigationDetailProps) {
  const { section, setSection } = useSectionRoute();
  const snapshot = useRunSnapshot(lens);
  const { job, batchId, batchSettings, batchFindings, missingSnapshot } = snapshot;
  return (
    <div>
      <section className="min-w-0 space-y-5">
        <div className="flex flex-wrap justify-between gap-3">
          <div className="min-w-0">
            <h2 className="text-lg font-semibold">{lens.settings.name}</h2>
            <InvestigationSummary lens={lens} />
          </div>
          {!readOnly && <InvestigationActions lens={lens} ready={ready} busy={busy} {...intents} />}
        </div>
        <RunReport
          job={job}
          findings={job?.findings}
          connected={connected}
          picker={<RunPicker lens={lens} job={job} />}
          onCancel={readOnly ? undefined : onCancelRun}
        />
        <Tabs value={section} onValueChange={setSection} key={lens.id}>
          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b">
            <TabsList variant="line">
              <TabsTrigger value="findings">Findings</TabsTrigger>
              <TabsTrigger value="checks">Criteria</TabsTrigger>
              <TabsTrigger value="runs">Runs</TabsTrigger>
              <TabsTrigger value="activity">History</TabsTrigger>
            </TabsList>
          </div>
          {snapshot.error && (
            <p role="alert" className="text-sm text-destructive">
              Could not load this run.{" "}
              <Button variant="link" onClick={snapshot.refetch}>
                Retry
              </Button>
            </p>
          )}
          {missingSnapshot && section !== "activity" && (
            <p className="text-sm text-muted-foreground">
              This older batch predates saved result snapshots. Its findings remain available under All accumulated
              findings.
            </p>
          )}
          <FindingsTab lens={lens} job={job} findings={batchFindings}>
            <FindingPanel
              readOnly={readOnly}
              busy={busy}
              sampledRuns={job?.sample?.executions}
              onReview={onReviewFinding}
            />
          </FindingsTab>
          <CriteriaTab settings={batchSettings} readOnly={readOnly} onEditCriteria={intents.onEdit} />
          <RunsTab key={job?.id ?? batchId} lens={lens} job={job} />
          <HistoryTab lens={lens} />
        </Tabs>
      </section>
    </div>
  );
}
