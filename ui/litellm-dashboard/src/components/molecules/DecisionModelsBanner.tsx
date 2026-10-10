import React, { useMemo, useState } from "react";
import { ArrowRight, Sparkles, X } from "lucide-react";

import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import {
  DECISIONS_DOCS_URL,
  DECISIONS_PROVIDERS_DOCS_URL,
  SYSTEM_ONE_PLAYGROUND_ROUTE,
  buildDecisionCatalog,
  decisionProviderNames,
} from "@/lib/decisionModels";
import { uiHref } from "@/utils/uiHref";

import { Button } from "@/components/ui/button";

const STORAGE_KEY = "hideDecisionModelsBanner";

interface DecisionModelsBannerProps {
  onAddModel?: () => void;
}

const DecisionModelsBanner: React.FC<DecisionModelsBannerProps> = ({ onAddModel }) => {
  const [dismissed, setDismissed] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem(STORAGE_KEY) === "true";
    }
    return false;
  });
  const { data: costMap } = useModelCostMap(!dismissed, true);
  const providers = useMemo(() => decisionProviderNames(buildDecisionCatalog(costMap)), [costMap]);

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
          {providers.length > 0 && (
            <>
              Works with{" "}
              <a href={DECISIONS_PROVIDERS_DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline">
                {providers.length} providers
              </a>
              : {providers.join(", ")}.{" "}
            </>
          )}
          {onAddModel && (
            <>
              Search <code>decision</code> in Add Model to find them.{" "}
            </>
          )}
          Call them at <code>/v1/decisions</code> or <code>/v1/systemone</code>, or try them in the System One
          playground.{" "}
          <a href={DECISIONS_DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline">
            How to call them
          </a>
        </p>
      </div>
      {onAddModel && (
        <Button type="button" variant="outline" className="shrink-0" onClick={onAddModel}>
          Add a decision model
        </Button>
      )}
      <Button
        className="shrink-0"
        nativeButton={false}
        role="link"
        render={<a href={uiHref(SYSTEM_ONE_PLAYGROUND_ROUTE)} />}
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
