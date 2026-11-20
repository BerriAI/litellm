import React, { useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { ArrowRight, Sparkles, X } from "lucide-react";

import { DECISIONS_DOCS_URL, DECISIONS_PROVIDERS_DOCS_URL, SYSTEM_ONE_PLAYGROUND_ROUTE } from "@/lib/decisionModels";
import { uiHref } from "@/utils/uiHref";

import { Button } from "@/components/ui/button";

const STORAGE_KEY = "hideDecisionModelsBanner";

const DecisionModelsBanner: React.FC = () => {
  const { t } = useTranslation();
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
    <div className="relative flex flex-wrap items-center gap-3 rounded-lg border bg-muted/40 p-4">
      <div className="flex size-8 shrink-0 items-center justify-center rounded-full border bg-background">
        <Sparkles className="size-4 text-muted-foreground" />
      </div>
      <div className="min-w-0 basis-48 flex-1 pr-6">
        <h4 className="m-0 text-sm font-semibold text-foreground">{t("decisionModelsBanner.title")}</h4>
        <p className="m-0 mt-0.5 text-xs text-muted-foreground">
          <Trans
            i18nKey="decisionModelsBanner.callInstructions"
            components={{ decisions: <code />, systemOne: <code /> }}
          />{" "}
          <a href={DECISIONS_DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline">
            {t("decisionModelsBanner.howToCall")}
          </a>
        </p>
      </div>
      <div className="flex max-w-full flex-wrap items-center gap-2 pr-4">
        <Button
          size="sm"
          variant="outline"
          className="shrink-0"
          nativeButton={false}
          role="link"
          render={<a href={DECISIONS_PROVIDERS_DOCS_URL} target="_blank" rel="noopener noreferrer" />}
        >
          {t("decisionModelsBanner.supportedProviders")}
        </Button>
        <Button
          size="sm"
          className="shrink-0"
          nativeButton={false}
          role="link"
          render={<a href={uiHref(SYSTEM_ONE_PLAYGROUND_ROUTE)} />}
        >
          {t("decisionModelsBanner.tryDecisionModels")}
          <ArrowRight />
        </Button>
      </div>
      <Button
        type="button"
        variant="ghost"
        size="icon-sm"
        onClick={() => {
          setDismissed(true);
          localStorage.setItem(STORAGE_KEY, "true");
        }}
        className="absolute top-3 right-3 shrink-0"
        aria-label={t("decisionModelsBanner.dismiss")}
      >
        <X />
      </Button>
    </div>
  );
};

export default DecisionModelsBanner;
