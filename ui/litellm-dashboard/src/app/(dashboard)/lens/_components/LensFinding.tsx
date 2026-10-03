import { ArrowUpRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet";
import { LensIssueBrief } from "./LensIssueBrief";
import { evidenceTarget, runTime, type Finding, type Sample } from "./lensData";

export function LensFinding({
  finding,
  sampledRuns,
  readOnly,
  reason,
  busy,
  onClose,
  onReason,
  onEvidence,
  changeFinding,
}: {
  finding?: Finding;
  sampledRuns: Sample["executions"];
  readOnly: boolean;
  reason: string;
  busy: boolean;
  onClose: () => void;
  onReason: (reason: string) => void;
  onEvidence: (evidence: { id: string; span: string }) => void;
  changeFinding: (status: Finding["status"]) => Promise<void>;
}) {
  const evidenceGroups = finding
    ? [...new Set(finding.evidence.map((e) => e.execution_id))].map((id) => ({
        id,
        run: sampledRuns.find((r) => r.id === id),
        quotes: finding.evidence.filter((e) => e.execution_id === id),
      }))
    : [];
  return (
    <Sheet
      open={!!finding}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <SheetContent className="overflow-y-auto data-[side=right]:sm:max-w-2xl">
        {finding && (
          <>
            <SheetHeader>
              <SheetTitle className="pr-8 text-xl leading-snug">{finding.title}</SheetTitle>
              <SheetDescription>
                {finding.kind === "issue" ? `${finding.priority} priority` : "Pattern"} ·{" "}
                {finding.occurrences?.length ?? 0} linked {finding.occurrences?.length === 1 ? "run" : "runs"}
              </SheetDescription>
            </SheetHeader>
            <div className="space-y-6 p-4">
              {finding.brief ? (
                <LensIssueBrief title={finding.title} brief={finding.brief} />
              ) : (
                <>
                  <div>
                    <p className="mb-2 text-sm font-medium">What happened</p>
                    <p className="text-sm leading-6 whitespace-pre-wrap">{finding.description}</p>
                  </div>
                  {finding.suggestion && (
                    <div className="border-y py-4">
                      <p className="text-sm font-medium">What to do next</p>
                      <p className="mt-2 text-sm leading-6">{finding.suggestion}</p>
                    </div>
                  )}
                </>
              )}
              {finding.limitation && (
                <details className="text-sm">
                  <summary className="cursor-pointer font-medium">Evidence limits</summary>
                  <p className="mt-3 leading-6 text-muted-foreground">{finding.limitation}</p>
                </details>
              )}
              <div>
                <p className="text-sm font-medium">Evidence by run</p>
                <p className="mt-1 mb-3 text-xs text-muted-foreground">
                  Exact quotes from the recorded activity. Counterexamples are labeled separately from supporting
                  evidence.
                </p>
                <div className="space-y-2">
                  {evidenceGroups.map((group) => (
                    <details key={group.id} className="rounded-lg border p-3">
                      <summary className="cursor-pointer text-sm font-medium">
                        {group.run?.name ?? evidenceTarget(group.id)?.id.slice(0, 12) ?? "Recorded run"}
                        <span className="ml-2 text-xs font-normal text-muted-foreground">
                          {group.quotes.length} {group.quotes.length === 1 ? "quote" : "quotes"}
                          {group.run ? ` · ${runTime(group.run.start_time)}` : ""}
                        </span>
                      </summary>
                      <div className="mt-3 space-y-3">
                        {group.quotes.map((e, i) => (
                          <div key={`${e.span_id}-${i}`} className="rounded-md bg-muted/40 p-3">
                            {e.role === "counterexample" && (
                              <p className="mb-1 text-xs font-medium text-muted-foreground">Counterexample</p>
                            )}
                            <blockquote className="text-xs leading-5 whitespace-pre-wrap break-words">
                              {e.quote}
                            </blockquote>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="mt-2"
                              onClick={() => {
                                onEvidence({ id: e.execution_id, span: e.span_id });
                              }}
                            >
                              {evidenceTarget(e.execution_id)?.source === "traces"
                                ? "Open original step"
                                : "Open request"}
                              <ArrowUpRight className="size-3" />
                            </Button>
                          </div>
                        ))}
                      </div>
                    </details>
                  ))}
                </div>
              </div>
              {!readOnly && (
                <div className="space-y-3 border-t pt-4">
                  <label className="grid gap-2 text-sm">
                    What should Lens remember?
                    <Textarea
                      value={reason}
                      onChange={(e) => onReason(e.target.value)}
                      placeholder="What should Lens know about this finding?"
                    />
                  </label>
                  <p className="text-xs text-muted-foreground">Your explanation informs future scans of this Lens.</p>
                  <div className="flex flex-wrap gap-2">
                    {finding.kind === "issue" && (
                      <Button
                        disabled={busy}
                        onClick={() => changeFinding(finding.status === "resolved" ? "open" : "resolved")}
                      >
                        {finding.status === "resolved" ? "Reopen" : "Mark resolved"}
                      </Button>
                    )}
                    <Button disabled={busy} variant="outline" onClick={() => changeFinding("dismissed")}>
                      This is expected
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}
