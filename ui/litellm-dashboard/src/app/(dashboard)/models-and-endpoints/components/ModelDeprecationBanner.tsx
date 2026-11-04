"use client";

import { TriangleAlert } from "lucide-react";
import { useState } from "react";
import { useModelDeprecations } from "@/app/(dashboard)/hooks/models/useModelDeprecations";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { uiHref } from "@/utils/uiHref";
import { deprecationBannerTitle, deprecationEntryText } from "./deprecationBannerText";

const ModelDeprecationBanner = () => {
  const { data, isError, isLoading } = useModelDeprecations();
  const [isExpanded, setIsExpanded] = useState(false);

  if (isLoading || isError || !data) {
    return null;
  }

  const deprecated = data.deprecated ?? [];
  const imminent = data.imminent ?? [];

  if (deprecated.length === 0 && imminent.length === 0) {
    return null;
  }

  const entries = [...deprecated, ...imminent];
  const visibleEntries = isExpanded ? entries : entries.slice(0, 5);

  return (
    <Alert variant={deprecated.length > 0 ? "destructive" : "default"} className="mb-4">
      <TriangleAlert />
      <AlertTitle>{deprecationBannerTitle(deprecated.length, imminent.length, data.warn_within_days)}</AlertTitle>
      <AlertDescription>
        <ul className="list-disc pl-5">
          {visibleEntries.map((info, index) => (
            <li key={`${info.model_name}-${info.deprecation_date}-${index}`}>{deprecationEntryText(info)}</li>
          ))}
        </ul>
        {entries.length > 5 && (
          <button
            type="button"
            className="mt-2 underline"
            aria-expanded={isExpanded}
            onClick={() => setIsExpanded((expanded) => !expanded)}
          >
            {isExpanded ? "Show fewer" : `Show all ${entries.length}`}
          </button>
        )}
        <a
          href={`${uiHref("logging-and-alerts")}?tab=alerting-types`}
          className="mt-2 inline-block font-medium underline"
        >
          Get Slack alerts for deprecations
        </a>
      </AlertDescription>
    </Alert>
  );
};

export default ModelDeprecationBanner;
