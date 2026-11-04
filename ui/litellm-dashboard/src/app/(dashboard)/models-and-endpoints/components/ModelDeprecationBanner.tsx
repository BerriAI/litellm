"use client";

import { TriangleAlert, X } from "lucide-react";
import { useState } from "react";
import { useModelDeprecations } from "@/app/(dashboard)/hooks/models/useModelDeprecations";
import { Alert, AlertAction, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { getLocalStorageItem, setLocalStorageItem } from "@/utils/localStorageUtils";
import { uiHref } from "@/utils/uiHref";
import {
  dedupeDeprecationEntries,
  deprecationBannerFingerprint,
  deprecationBannerTitle,
  deprecationEntryText,
} from "./deprecationBannerText";

const STORAGE_KEY = "modelDeprecationBannerDismissed";

const ModelDeprecationBanner = () => {
  const { data, isError, isLoading } = useModelDeprecations();
  const [isExpanded, setIsExpanded] = useState(false);
  const [dismissedFingerprint, setDismissedFingerprint] = useState<string | null>(() =>
    getLocalStorageItem(STORAGE_KEY),
  );

  if (isLoading || isError || !data) {
    return null;
  }

  const deprecated = dedupeDeprecationEntries(data.deprecated ?? []);
  const imminent = dedupeDeprecationEntries(data.imminent ?? []);

  if (deprecated.length === 0 && imminent.length === 0) {
    return null;
  }

  const entries = [...deprecated, ...imminent];
  const fingerprint = deprecationBannerFingerprint(entries);
  if (dismissedFingerprint === fingerprint) {
    return null;
  }

  const visibleEntries = isExpanded ? entries : entries.slice(0, 5);

  return (
    <Alert variant={deprecated.length > 0 ? "destructive" : "default"} className="mb-4">
      <TriangleAlert />
      <AlertAction>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          onClick={() => {
            setDismissedFingerprint(fingerprint);
            setLocalStorageItem(STORAGE_KEY, fingerprint);
          }}
          aria-label="Dismiss deprecation warning"
        >
          <X />
        </Button>
      </AlertAction>
      <AlertTitle>{deprecationBannerTitle(deprecated.length, imminent.length, data.warn_within_days)}</AlertTitle>
      <AlertDescription>
        <ul className="list-disc pl-5">
          {visibleEntries.map((info, index) => (
            <li key={`${info.model_name}-${info.deprecation_date}-${index}`}>{deprecationEntryText(info)}</li>
          ))}
        </ul>
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1">
          {entries.length > 5 && (
            <button
              type="button"
              className="underline"
              aria-expanded={isExpanded}
              onClick={() => setIsExpanded((expanded) => !expanded)}
            >
              {isExpanded ? "Show fewer" : `Show all ${entries.length}`}
            </button>
          )}
          <a href={`${uiHref("logging-and-alerts")}?tab=alerting-types`} className="font-medium underline">
            Get Slack alerts for deprecations
          </a>
        </div>
      </AlertDescription>
    </Alert>
  );
};

export default ModelDeprecationBanner;
