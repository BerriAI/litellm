"use client";

import { useState } from "react";
import { ChevronRight, ClipboardCopy, X } from "lucide-react";

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
import { PriorityPill } from "./PriorityMark";
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
    <div className="sticky top-0 z-raised flex h-11 items-center justify-between gap-2 bg-background/95 px-4 backdrop-blur">
      <p className="flex min-w-0 items-center gap-2 font-mono text-xs text-muted-foreground">
        <span className="truncate" title={finding.id}>
          {finding.id.slice(0, 8)}
        </span>
        <span aria-hidden="true">·</span>
        <span className="whitespace-nowrap tabular-nums" title={runTime(finding.last_seen)}>
          {agoLabel(Date.parse(finding.last_seen), now)}
        </span>
      </p>
      <div className="flex shrink-0 gap-1.5">
        <Button
          variant="outline"
          size="xs"
          className="enabled:active:scale-[0.96]"
          onClick={() => void copyToClipboard(findingMarkdown(finding), "Copied for agent")}
        >
          <ClipboardCopy />
          Copy for agent
        </Button>
        {onClose && (
          <Button
            variant="ghost"
            size="xs"
            className="enabled:active:scale-[0.96]"
            aria-label="Close finding (Esc)"
            onClick={onClose}
          >
            <X />
            Close
          </Button>
        )}
      </div>
    </div>
  );
}

function Disclosure({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="group rounded-lg">
      <summary className="flex cursor-pointer list-none items-center gap-1.5 text-sm font-semibold [&::-webkit-details-marker]:hidden">
        <ChevronRight
          aria-hidden="true"
          className="size-4 text-muted-foreground transition-[rotate] duration-150 group-open:rotate-90 motion-reduce:transition-none"
        />
        {title}
      </summary>
      <div className="mt-3">{children}</div>
    </details>
  );
}

function ProseSection({ title, children }: { title: string; children: string }) {
  return (
    <section>
      <h2 className="mb-1.5 text-sm font-semibold">{title}</h2>
      <p className="max-w-[70ch] text-sm leading-relaxed text-pretty whitespace-pre-wrap text-foreground/85">
        {children}
      </p>
    </section>
  );
}

const FIELD = /^(Input|Output|Status|Error)\s*:/gm;
const FIELD_LABEL: Readonly<Record<string, string>> = {
  "Input,Output": "Call and result",
  Input: "Call input",
  Output: "Returned output",
  Status: "Span status",
  Error: "Error",
};

function quoteLabel(quote: Quote, isTrace: boolean): string {
  if (quote.role === "counterexample") return "Counterexample";
  const fields = [...new Set(Array.from(quote.quote.matchAll(FIELD), (m) => m[1]))].join(",");
  return FIELD_LABEL[fields] ?? (isTrace ? "Trace step" : "Logged request");
}

const MARK = {
  support: "rounded-sm bg-finding-quote px-0.5 text-inherit",
  counterexample: "rounded-sm bg-success/20 px-0.5 text-inherit",
} as const;

function QuoteCard({ quote, onOpen }: { quote: Quote; onOpen: () => void }) {
  const isTrace = evidenceTarget(quote.execution_id)?.source === "traces";
  return (
    <div className="-mx-1 rounded-lg border border-transparent bg-background p-2.5 shadow-finding-ring">
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <span className="min-w-0 truncate text-sm font-medium">{quoteLabel(quote, isTrace)}</span>
        <Button variant="outline" size="xs" className="enabled:active:scale-[0.96]" onClick={onOpen}>
          {isTrace ? "View span" : "View request"}
        </Button>
      </div>
      <button
        type="button"
        tabIndex={-1}
        title="Open this evidence"
        onClick={onOpen}
        className="block w-full overflow-x-auto rounded-md bg-muted/60 py-1.5 text-left font-mono text-xs leading-5 transition-[background-color] duration-150 hover:bg-muted"
      >
        <code className="block border-l-2 border-warning/40 px-2.5 break-words whitespace-pre-wrap text-foreground/85">
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
  const traceId = evidenceTarget(group.id)?.id;
  const name = group.run?.name ?? (traceId ? `Trace ${traceId.slice(0, 8)}` : "Recorded run");
  return (
    <article aria-label={name} className="flex w-full flex-col items-start gap-3 rounded-xl bg-muted/50 p-3.5">
      <div className="flex w-full items-baseline justify-between gap-3">
        <h3 className="min-w-0 truncate text-sm font-medium" title={traceId}>
          {name}
        </h3>
        <span className="shrink-0 text-xs text-muted-foreground tabular-nums">
          {[group.run?.service, group.run && runTime(group.run.start_time)].filter(Boolean).join(" · ")}
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

const VISIBLE_EXAMPLES = 3;

function Examples({
  groups,
  onOpenEvidence,
}: {
  groups: readonly ExampleGroup[];
  onOpenEvidence: (e: EvidenceRef) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  if (groups.length === 0) return <p className="text-sm text-muted-foreground">No examples were recorded.</p>;
  const shown = expanded ? groups : groups.slice(0, VISIBLE_EXAMPLES);
  const hidden = groups.length - shown.length;
  return (
    <>
      {shown.map((group) => (
        <Example key={group.id} group={group} onOpenEvidence={onOpenEvidence} />
      ))}
      {hidden > 0 && (
        <Button
          variant="ghost"
          size="sm"
          className="self-start text-muted-foreground"
          onClick={() => setExpanded(true)}
        >
          Show {hidden} more {hidden === 1 ? "example" : "examples"}
        </Button>
      )}
    </>
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
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-7 px-6 pt-4 pb-16">
        <header className="flex flex-col gap-3">
          <h1 className="text-2xl leading-tight font-semibold tracking-tight text-balance">{finding.title}</h1>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs text-muted-foreground">
            {finding.kind === "issue" ? (
              <PriorityPill priority={finding.priority} />
            ) : (
              <span className="inline-flex h-5 items-center rounded-full bg-muted px-2 font-medium">Pattern</span>
            )}
            {agents.length > 0 && <span className="font-medium text-foreground">{agents.join(", ")}</span>}
            <span className="tabular-nums">
              {affected} affected {affected === 1 ? "trace" : "traces"}
            </span>
            {runs > 0 && (
              <span className="tabular-nums">
                Found across {runs} investigation {runs === 1 ? "run" : "runs"}
              </span>
            )}
          </div>
        </header>
        <ProseSection title="Summary">{finding.description}</ProseSection>
        {finding.suggestion && <ProseSection title="Suggested fix">{finding.suggestion}</ProseSection>}
        {finding.brief && (
          <Disclosure title="Issue brief and test cases">
            <IssueBrief title={finding.title} brief={finding.brief} />
          </Disclosure>
        )}
        {finding.limitation && (
          <Disclosure title="Evidence limits">
            <p className="max-w-[70ch] text-sm leading-relaxed text-pretty text-muted-foreground">
              {finding.limitation}
            </p>
          </Disclosure>
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
          <Examples groups={groups} onOpenEvidence={onOpenEvidence} />
        </section>
        {!readOnly && <ReviewForm finding={finding} busy={busy} onReview={onReview} />}
      </div>
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
