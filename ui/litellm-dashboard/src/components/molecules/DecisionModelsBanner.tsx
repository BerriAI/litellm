import React, { useState } from "react";
import { ArrowRight, Sparkles, X } from "lucide-react";

import { uiHref } from "@/utils/uiHref";

import { Button } from "@/components/ui/button";

const STORAGE_KEY = "hideDecisionModelsBanner";

const DecisionModelsBanner: React.FC = () => {
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
        <Sparkles className="size-4 text-muted-foreground" />
      </div>
      <div className="min-w-0 flex-1">
        <h4 className="m-0 text-sm font-semibold text-foreground">Decision models are now supported</h4>
        <p className="m-0 mt-0.5 text-xs text-muted-foreground">
          Explore decision models in LiteLLM. Try them in the System One playground.
        </p>
      </div>
      <Button
        className="shrink-0"
        nativeButton={false}
        role="link"
        render={<a href={uiHref("playground?tab=system-one")} />}
      >
        Try decision models
        <ArrowRight />
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

export default DecisionModelsBanner;
