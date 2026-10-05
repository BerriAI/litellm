"use client";

import React from "react";

import { extractErrorMessage } from "@/utils/errorUtils";
import { Button, buttonVariants } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { effortNote, estimateLabel, branchCostLabel } from "./roiCalculatorData";
import type { ROIIdentityMapUpdate, ROIPull, ROISummary } from "./roiCalculatorData";
import type { ROIPerson } from "./roiCalculatorData";

export type PersonMatchSelection = { person: ROIPerson; login: string };

export function PullReasoningDialog({
  pull,
  summary,
  onClose,
}: {
  pull: ROIPull | null;
  summary: ROISummary | null;
  onClose: () => void;
}) {
  const titleRef = React.useRef<HTMLHeadingElement>(null);
  return (
    <Dialog open={Boolean(pull)} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[85dvh] overflow-y-auto sm:max-w-2xl" initialFocus={titleRef}>
        {pull && (
          <>
            <DialogHeader>
              <DialogTitle ref={titleRef} tabIndex={-1} className="pr-8 text-lg leading-6">
                {pull.title}
              </DialogTitle>
              <DialogDescription>
                {pull.repo} #{pull.number} · {pull.login}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-3">
              <dl className="grid grid-cols-2 gap-4 rounded-lg border bg-muted/20 p-4">
                <div className="space-y-2">
                  <dt className="text-xs text-muted-foreground">Estimated effort</dt>
                  <dd className="text-2xl font-semibold tabular-nums">{estimateLabel(pull.estimate)}</dd>
                </div>
                <div className="space-y-2">
                  <dt className="text-xs text-muted-foreground">Recorded AI cost</dt>
                  <dd className="text-xl font-semibold tabular-nums">{branchCostLabel(pull)}</dd>
                  {pull.branch_cost?.status === "matched" && (
                    <dd className="text-xs text-muted-foreground">{pull.branch_cost.requests} requests</dd>
                  )}
                </div>
              </dl>
              <p className="text-xs leading-5 text-muted-foreground">
                {effortNote(pull.estimate.effort_basis ?? summary?.effort_basis)}
              </p>
            </div>
            <section>
              <h3 className="mb-2 font-medium">Reasoning</h3>
              <p className="whitespace-pre-wrap leading-relaxed">
                {pull.estimate.reasoning || "No estimate available."}
              </p>
            </section>
            <dl className="grid grid-cols-[auto_1fr] gap-x-5 gap-y-2 text-xs">
              <dt className="text-muted-foreground">Model</dt>
              <dd className="break-all">{pull.estimate.model || summary?.estimator_model}</dd>
              <dt className="text-muted-foreground">Merged</dt>
              <dd>{new Date(pull.merged_at).toLocaleDateString(undefined, { timeZone: "UTC" })}</dd>
              <dt className="text-muted-foreground">Email match</dt>
              <dd>{pull.email || "Not matched"}</dd>
            </dl>
            <section className="space-y-2 border-t pt-4 text-sm">
              <h3 className="font-medium">Track costs for this branch</h3>
              {pull.branch_cost?.status === "matched" && (
                <p className="text-muted-foreground">
                  {pull.branch_cost.spend?.toFixed(8)} USD across {pull.branch_cost.requests} requests
                </p>
              )}
              {pull.source_repo && pull.source_branch ? (
                <>
                  <p className="text-muted-foreground">Send both tags with each gateway request from this branch:</p>
                  <pre className="overflow-x-auto rounded-md bg-muted p-3 text-xs">
                    {JSON.stringify(
                      { metadata: { tags: [`repo:${pull.source_repo}`, `branch:${pull.source_branch}`] } },
                      null,
                      2,
                    )}
                  </pre>
                  <p className="text-xs text-muted-foreground">
                    Retained requests in the report’s UTC period. Branch names are case-sensitive. Reused branches
                    cannot be split between changes.
                  </p>
                </>
              ) : (
                <p className="text-muted-foreground">
                  The source repository or branch is unavailable. Sync again to refresh its metadata.
                </p>
              )}
            </section>
            {summary?.estimator_prompt && (
              <details className="rounded-md border p-3">
                <summary className="cursor-pointer font-medium">Estimator prompt</summary>
                <p className="mt-2 whitespace-pre-wrap text-sm">{summary.estimator_prompt}</p>
              </details>
            )}
            <DialogFooter>
              {pull.url && (
                <a
                  className={buttonVariants({ variant: "outline" })}
                  href={pull.url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  View on {summary?.source_provider === "gitlab" ? "GitLab" : "GitHub"}
                </a>
              )}
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

export function IdentityMatchDialog({
  selection,
  identityMap,
  gatewayEmails,
  onClose,
  onSave,
}: {
  selection: PersonMatchSelection | null;
  identityMap: Record<string, string>;
  gatewayEmails: string[];
  onClose: () => void;
  onSave: (payload: ROIIdentityMapUpdate) => Promise<void>;
}) {
  const [email, setEmail] = React.useState(() =>
    selection ? identityMap[selection.login.toLowerCase()] ?? selection.person.email ?? "" : "",
  );
  const [error, setError] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);
  const person = selection?.person ?? null;
  const login = selection?.login ?? "";
  const existingEmail = identityMap[login.toLowerCase()];

  const save = async (value: string | null) => {
    if (!login) return;
    try {
      setBusy(true);
      await onSave({ github_login: login, email: value });
      setError(null);
      onClose();
    } catch (reason) {
      setError(extractErrorMessage(reason));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={Boolean(person)} onOpenChange={(open) => !open && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Match email</DialogTitle>
          <DialogDescription>Link {login} to their gateway email. Manual matches take priority.</DialogDescription>
        </DialogHeader>
        <form
          className="space-y-4"
          onSubmit={(event) => {
            event.preventDefault();
            void save(email.trim());
          }}
        >
          <div className="grid gap-2">
            <Label htmlFor="roi-match-email">Gateway email</Label>
            <Input
              id="roi-match-email"
              autoComplete="off"
              list="roi-gateway-emails"
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
            />
          </div>
          <datalist id="roi-gateway-emails">
            {Array.from(new Set(gatewayEmails)).map((address) => (
              <option key={address} value={address} />
            ))}
          </datalist>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <DialogFooter>
            {existingEmail && (
              <Button disabled={busy} type="button" variant="outline" onClick={() => void save(null)}>
                Use automatic match
              </Button>
            )}
            <Button disabled={busy || !email.trim()} type="submit">
              {busy ? "Saving…" : "Save match"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
