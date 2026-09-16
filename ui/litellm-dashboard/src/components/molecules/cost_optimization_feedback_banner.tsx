import React, { useState } from "react";
import { ExternalLink, MessageSquare, X } from "lucide-react";

import { Button } from "@/components/ui/button";

const STORAGE_KEY = "hideCostOptimizationFeedbackBanner";
const DISCUSSION_URL = "https://github.com/BerriAI/litellm/discussions/32172";

const CostOptimizationFeedbackBanner: React.FC = () => {
  const [dismissed, setDismissed] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem(STORAGE_KEY) === "true";
    }
    return false;
  });

  if (dismissed) {
    return null;
  }

  return (
    <div className="mb-4 flex items-center gap-4 rounded-lg border bg-muted/40 px-4 py-3">
      <div className="flex size-10 shrink-0 items-center justify-center rounded-full border bg-background">
        <MessageSquare className="size-4 text-muted-foreground" />
      </div>
      <div className="min-w-0 flex-1">
        <h4 className="m-0 text-sm font-semibold text-foreground">Help shape cost optimization</h4>
        <p className="m-0 mt-0.5 text-xs text-muted-foreground">
          We&apos;re collecting suggestions for cost optimization improvements across routing, budgets, and more. Let us
          know what you&apos;d like to see.
        </p>
      </div>
      <Button
        className="shrink-0"
        nativeButton={false}
        render={<a href={DISCUSSION_URL} target="_blank" rel="noopener noreferrer" />}
      >
        Share Feedback
        <ExternalLink />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon-sm"
        onClick={() => {
          setDismissed(true);
          localStorage.setItem(STORAGE_KEY, "true");
        }}
        className="shrink-0"
        aria-label="Dismiss banner"
      >
        <X />
      </Button>
    </div>
  );
};

export default CostOptimizationFeedbackBanner;
