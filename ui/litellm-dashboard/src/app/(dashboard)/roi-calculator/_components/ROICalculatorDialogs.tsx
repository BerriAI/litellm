"use client";

import React from "react";

import { extractErrorMessage } from "@/utils/errorUtils";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { effortNote, estimateLabel } from "./roiCalculatorData";
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
  return (
    <Dialog open={Boolean(pull)} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-h-[85dvh] overflow-y-auto sm:max-w-xl">
        {pull && (
          <>
            <DialogHeader>
              <DialogTitle>{pull.title}</DialogTitle>
              <DialogDescription>{pull.repo} #{pull.number} · {pull.login}</DialogDescription>
            </DialogHeader>
            <div>
              <p className="text-sm text-muted-foreground">Estimated engineering hours</p>
              <p className="mt-2 text-3xl font-semibold tabular-nums">{estimateLabel(pull.estimate)}</p>
              <p className="mt-2 text-xs text-muted-foreground">
                {effortNote(pull.estimate.effort_basis ?? summary?.effort_basis)}
              </p>
              {pull.estimate.evidence_source === "pr_metadata" && (
                <p className="mt-1 text-xs text-muted-foreground">
                  Based on PR descriptions, file change counts, and commit metadata.
                </p>
              )}
            </div>
            <section>
              <h3 className="mb-2 font-medium">Reasoning</h3>
              <p className="whitespace-pre-wrap leading-relaxed">{pull.estimate.reasoning || "No estimate available."}</p>
            </section>
            <dl className="grid grid-cols-[auto_1fr] gap-x-5 gap-y-2 text-xs">
              <dt className="text-muted-foreground">Model</dt>
              <dd className="break-all">{pull.estimate.model || summary?.estimator_model}</dd>
              <dt className="text-muted-foreground">Merged</dt>
              <dd>{new Date(pull.merged_at).toLocaleDateString(undefined, { timeZone: "UTC" })}</dd>
              <dt className="text-muted-foreground">Email match</dt>
              <dd>{pull.email || "Not matched"}</dd>
            </dl>
            {summary?.estimator_prompt && (
              <details className="rounded-md border p-3">
                <summary className="cursor-pointer font-medium">Estimator prompt</summary>
                <p className="mt-2 whitespace-pre-wrap text-sm">{summary.estimator_prompt}</p>
              </details>
            )}
            <DialogFooter>
              <Button variant="outline" render={<a href={pull.url} target="_blank" rel="noopener noreferrer" />}>
                View on GitHub
              </Button>
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
  onClose,
  onSave,
}: {
  selection: PersonMatchSelection | null;
  identityMap: Record<string, string>;
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
              type="email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
            />
          </div>
          {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
          <DialogFooter>
            {existingEmail && (
              <Button disabled={busy} type="button" variant="outline" onClick={() => void save(null)}>
                Use automatic match
              </Button>
            )}
            <Button disabled={busy || !email.trim()} type="submit">{busy ? "Saving…" : "Save match"}</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
