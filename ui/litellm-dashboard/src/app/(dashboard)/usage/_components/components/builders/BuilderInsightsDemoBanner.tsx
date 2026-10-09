"use client";

import { Info } from "lucide-react";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";

const CONNECTION_STEPS = [
  "Connect your GitHub org once (GitHub App or one admin token). No per user tokens.",
  "Map each LiteLLM user email to a GitHub login.",
  "Verdicts refresh nightly from SpendLogs and merged PRs.",
];

export function BuilderInsightsDemoBanner() {
  const [open, setOpen] = useState(false);

  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-card px-4 py-3">
        <div className="flex min-w-0 items-center gap-2.5 text-sm text-muted-foreground">
          <Info aria-hidden="true" className="size-4 shrink-0" />
          <p>Demo data. Builder Insights joins SpendLogs with merged PRs from GitHub to show your team.</p>
        </div>
        <Button type="button" variant="outline" size="sm" onClick={() => setOpen(true)}>
          How to connect
        </Button>
      </div>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>How to connect</DialogTitle>
            <DialogDescription>Connect GitHub data to see Builder Insights for your team.</DialogDescription>
          </DialogHeader>
          <ol className="grid gap-3">
            {CONNECTION_STEPS.map((step, index) => (
              <li key={step} className="flex gap-3 text-sm leading-6">
                <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-muted text-xs tabular-nums text-muted-foreground">
                  {index + 1}
                </span>
                <span>{step}</span>
              </li>
            ))}
          </ol>
        </DialogContent>
      </Dialog>
    </>
  );
}
