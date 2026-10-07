"use client";

import { useState } from "react";
import { ClipboardCopy, X } from "lucide-react";

import { Inspector } from "@/components/shared/Inspector";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useNow } from "@/hooks/useNow";
import { copyToClipboard } from "@/utils/dataUtils";

import { AddToDatasetButton } from "../datasets/AddToDatasetDialog";
import { evidenceTarget, findingMarkdown } from "../model/findings";
import { findingFrequency } from "../model/frequency";
import { agoLabel, runTime } from "../model/format";
import { findingAgents, findingKey, type OwnedFinding, sampledExecutions } from "../model/inbox";
import type { Finding, Sample } from "../model/types";
import { EvidenceView } from "./Evidence";
import { FrequencyCard } from "./FrequencyCard";
import { IssueBrief } from "./IssueBrief";
import { type EvidenceRef, useEvidenceRoute } from "../route";

export const ownedFindingKey = (owned: OwnedFinding): string => findingKey(owned.lens, owned.finding);

type Quote = Finding["evidence"][number];

const SECTION_LABEL = "text-xs font-medium text-muted-foreground";

export interface FindingDetailsProps {
  readonly finding: Finding;
  readonly lensId?: string;
  readonly agents?: readonly string[];
  readonly sampledRuns: Sample["executions"];
  readonly readOnly: boolean;
  readonly busy: boolean;
  readonly onOpenEvidence: (evidence: EvidenceRef) => void;
  readonly onReview: (status: Finding["status"], reason: string) => void;
  readonly onClose?: () => void;
}

function TopBar({ finding, onClose }: Pick<FindingDetailsProps, "finding" | "onClose">) {
  const now = useNow(30000);
  return (
    <div className="sticky top-0 z-raised flex items-center justify-between gap-2 bg-background px-3 py-2">
      <p className="flex min-w-0 items-center gap-2 font-mono text-xs text-muted-foreground">
        <span className="truncate" title={finding.id}>
          {finding.id.slice(0, 8)}
        </span>
        <span aria-hidden="true">·</span>
        <span className="whitespace-nowrap tabular-nums" title={runTime(finding.last_seen)}>
          {agoLabel(Date.parse(finding.last_seen), now)}
        </span>
      </p>
      <div className="flex shrink-0 gap-2">
        <Button
          variant="outline"
          size="sm"
          onClick={() => void copyToClipboard(findingMarkdown(finding), "Copied for agent")}
        >
          <ClipboardCopy className="size-3.5" />
          Copy for agent
        </Button>
        {onClose && (
          <Button variant="outline" size="sm" aria-label="Close finding (Esc)" onClick={onClose}>
            <X className="size-3.5" />
            Close
          </Button>
        )}
      </div>
    </div>
  );
}

function ProseSection({ title, children }: { title: string; children: string }) {
  return (
    <section>
      <h2 className="mb-2 text-base font-medium">{title}</h2>
      <p className="text-sm leading-relaxed whitespace-pre-wrap text-foreground/90">{children}</p>
    </section>
  );
}

function quoteLabel(quote: Quote, isTrace: boolean): string {
  if (quote.role === "counterexample") return "Counterexample";
  return isTrace ? "Trace step" : "Logged request";
}

const MARK = {
  support: "rounded-sm bg-finding-quote px-0.5 text-inherit",
  counterexample: "rounded-sm bg-success/20 px-0.5 text-inherit",
} as const;

function QuoteCard({ quote, onOpen }: { quote: Quote; onOpen: () => void }) {
  const isTrace = evidenceTarget(quote.execution_id)?.source === "traces";
  return (
    <div className="-mx-1 rounded-md border bg-background p-2">
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="min-w-0 truncate text-sm font-medium">{quoteLabel(quote, isTrace)}</span>
        <Button variant="outline" size="xs" onClick={onOpen}>
          {isTrace ? "View span" : "View request"}
        </Button>
      </div>
      <button
        type="button"
        tabIndex={-1}
        title="Open this evidence"
        onClick={onOpen}
        className="mt-1 block w-full overflow-x-auto bg-muted/70 py-1 text-left font-mono text-xs leading-5 hover:bg-muted"
      >
        <code className="block border-l border-border px-2 py-0.5 break-words whitespace-pre-wrap text-foreground/80">
          <mark className={MARK[quote.role]}>{quote.quote}</mark>
        </code>
      </button>
    </div>
  );
}

function EvidenceRail({ children }: { children: React.ReactNode }) {
  return (
    <>
      <div className="relative flex gap-2">
        <div className="relative flex w-3 shrink-0 flex-col items-center">
          <div className="flex h-5 w-full items-center justify-center">
            <div className="size-1 rounded-full bg-border" />
          </div>
          <div aria-hidden="true" className="h-2 w-px bg-border" />
        </div>
        <div className="min-w-0 flex-1 text-xs leading-5 font-medium text-muted-foreground">Evidence</div>
      </div>
      <div className="relative">
        <div aria-hidden="true" className="absolute inset-y-0 left-[5px] w-px bg-border" />
        <div className="relative z-raised space-y-3">{children}</div>
      </div>
    </>
  );
}

interface ExampleGroup {
  readonly id: string;
  readonly run: Sample["executions"][number] | undefined;
  readonly quotes: readonly Quote[];
}

function Example({ group, onOpenEvidence }: { group: ExampleGroup; onOpenEvidence: (e: EvidenceRef) => void }) {
  const name = group.run?.name ?? evidenceTarget(group.id)?.id.slice(0, 12) ?? "Recorded run";
  return (
    <article aria-label={name} className="flex w-full flex-col items-start gap-3 rounded-md bg-muted/40 p-3">
      <div className="flex w-full items-baseline justify-between gap-3">
        <h3 className="min-w-0 truncate text-sm font-medium">{name}</h3>
        <span className="shrink-0 text-xs text-muted-foreground">
          {group.run?.service && `${group.run.service} · `}
          {group.run ? runTime(group.run.start_time) : ""}
        </span>
      </div>
      <div className="w-full">
        {group.quotes.length === 0 ? (
          <div className="flex items-center justify-between gap-3">
            <p className="text-sm text-muted-foreground">No quote was retained for this trace.</p>
            <Button variant="outline" size="xs" onClick={() => onOpenEvidence({ id: group.id, span: "" })}>
              View trace
            </Button>
          </div>
        ) : (
          <EvidenceRail>
            {group.quotes.map((quote, i) => (
              <QuoteCard
                key={`${quote.span_id}-${i}`}
                quote={quote}
                onOpen={() => onOpenEvidence({ id: quote.execution_id, span: quote.span_id })}
              />
            ))}
          </EvidenceRail>
        )}
      </div>
    </article>
  );
}

function ReviewForm({ finding, busy, onReview }: Pick<FindingDetailsProps, "finding" | "busy" | "onReview">) {
  const [reason, setReason] = useState(finding.reason ?? "");
  return (
    <section className="space-y-3 border-t pt-6">
      <label className="grid gap-2 text-sm font-medium">
        What should Lens remember?
        <Textarea
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          placeholder="What should Lens know about this finding?"
        />
      </label>
      <p className="text-xs text-muted-foreground">Your explanation informs future scans of this Lens.</p>
      <div className="flex flex-wrap gap-2">
        {finding.kind === "issue" && (
          <Button disabled={busy} onClick={() => onReview(finding.status === "resolved" ? "open" : "resolved", reason)}>
            {finding.status === "resolved" ? "Reopen" : "Mark resolved"}
          </Button>
        )}
        <Button disabled={busy} variant="outline" onClick={() => onReview("dismissed", reason)}>
          This is expected
        </Button>
      </div>
    </section>
  );
}

export function FindingDetails({
  finding,
  lensId,
  agents = [],
  sampledRuns,
  readOnly,
  busy,
  onOpenEvidence,
  onReview,
  onClose,
}: FindingDetailsProps) {
  const groups: ExampleGroup[] = [
    ...new Set([...finding.evidence.map((e) => e.execution_id), ...finding.occurrences]),
  ].map((id) => ({
    id,
    run: sampledRuns.find((r) => r.id === id),
    quotes: finding.evidence.filter((e) => e.execution_id === id),
  }));
  const affected = finding.occurrences.length;
  const runs = finding.investigation_runs?.length ?? 0;
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <TopBar finding={finding} onClose={onClose} />
      <article className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-6 pt-6 pb-12">
        <header className="flex flex-col gap-2">
          <h1 className="text-2xl font-semibold text-balance">{finding.title}</h1>
          <p className="text-xs text-muted-foreground">
            {agents.length > 0 && <span className="font-medium text-foreground">{agents.join(", ")} · </span>}
            {finding.kind === "issue" ? `${finding.priority} priority` : "Pattern"} · {affected} affected{" "}
            {affected === 1 ? "trace" : "traces"}
            {runs > 0 && ` · Found across ${runs} investigation ${runs === 1 ? "run" : "runs"}`}
          </p>
        </header>
        <ProseSection title="Summary">{finding.description}</ProseSection>
        {finding.suggestion && <ProseSection title="Suggested fix">{finding.suggestion}</ProseSection>}
        {finding.brief && (
          <details className="group">
            <summary className="mb-3 cursor-pointer text-base font-medium">Issue brief and test cases</summary>
            <IssueBrief title={finding.title} brief={finding.brief} />
          </details>
        )}
        {finding.limitation && (
          <details className="text-sm">
            <summary className="cursor-pointer font-medium">Evidence limits</summary>
            <p className="mt-3 leading-6 text-muted-foreground">{finding.limitation}</p>
          </details>
        )}
        <section>
          <h2 className={`mb-2 ${SECTION_LABEL}`}>Monitors</h2>
          <FrequencyCard frequency={findingFrequency(finding.occurrences, sampledRuns)} />
        </section>
        <section className="flex flex-col gap-2">
          <div className="flex items-center justify-between gap-2">
            <h2 className={SECTION_LABEL}>Examples</h2>
            {lensId && finding.evidence.length > 0 && (
              <AddToDatasetButton
                sources={[{ kind: "finding", lens_id: lensId, finding_ids: [finding.id] }]}
                agentName={agents[0]}
                label="Add evidence to dataset"
              />
            )}
          </div>
          {groups.length === 0 && <p className="text-sm text-muted-foreground">No examples were recorded.</p>}
          {groups.map((group) => (
            <Example key={group.id} group={group} onOpenEvidence={onOpenEvidence} />
          ))}
        </section>
        {!readOnly && <ReviewForm finding={finding} busy={busy} onReview={onReview} />}
      </article>
    </div>
  );
}

export interface FindingPanelProps {
  readonly readOnly: boolean;
  readonly busy: boolean;
  /** Runs to name evidence by; defaults to every run the owning investigation has sampled. */
  readonly sampledRuns?: Sample["executions"];
  readonly onReview: (owned: OwnedFinding, status: Finding["status"], reason: string) => void;
}

/**
 * A finding in the side panel. Opening a quote's original step or request stacks it over the finding, which
 * stays mounted so a feedback draft survives the round trip.
 */
export function FindingPanelBody({
  owned,
  readOnly,
  busy,
  sampledRuns,
  onReview,
}: FindingPanelProps & { readonly owned: OwnedFinding }) {
  const { evidence, setEvidence } = useEvidenceRoute();
  return (
    <>
      <div hidden={evidence !== null} className={evidence ? undefined : "flex min-h-0 flex-1 flex-col"}>
        <FindingDetails
          key={ownedFindingKey(owned)}
          finding={owned.finding}
          lensId={owned.lens.id}
          agents={findingAgents(owned.lens, owned.finding)}
          sampledRuns={sampledRuns ?? sampledExecutions(owned.lens)}
          readOnly={readOnly}
          busy={busy}
          onOpenEvidence={setEvidence}
          onReview={(status, reason) => onReview(owned, status, reason)}
        />
      </div>
      {evidence && (
        <EvidenceView
          lensId={owned.lens.id}
          evidence={evidence}
          backLabel="Back to finding"
          onBack={() => setEvidence(null)}
        />
      )}
    </>
  );
}

/** The side panel for whichever finding the surrounding Inspector has open. */
export function FindingPanel(props: FindingPanelProps) {
  return (
    <Inspector.Panel label="Finding details" testId="finding-panel">
      {(owned: OwnedFinding) => <FindingPanelBody {...props} owned={owned} />}
    </Inspector.Panel>
  );
}
