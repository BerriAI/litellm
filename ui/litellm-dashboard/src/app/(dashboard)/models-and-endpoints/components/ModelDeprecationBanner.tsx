"use client";

import { TriangleAlert } from "lucide-react";
import { useModelDeprecations } from "@/app/(dashboard)/hooks/models/useModelDeprecations";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { uiHref } from "@/utils/uiHref";
import { deprecationBannerTitle, deprecationEntryText } from "./modelDeprecationBanner";

const ModelDeprecationBanner = () => {
  const { data, isError, isLoading } = useModelDeprecations();

  if (isLoading || isError || !data) {
    return null;
  }

  const deprecated = data.deprecated ?? [];
  const imminent = data.imminent ?? [];

  if (deprecated.length === 0 && imminent.length === 0) {
    return null;
  }

  const entries = [...deprecated, ...imminent];
  const visibleEntries = entries.slice(0, 5);
  const remainingCount = entries.length - visibleEntries.length;

  return (
    <Alert variant={deprecated.length > 0 ? "destructive" : "default"} className="mb-4">
      <TriangleAlert />
      <AlertTitle>{deprecationBannerTitle(deprecated.length, imminent.length, data.warn_within_days)}</AlertTitle>
      <AlertDescription>
        <ul className="list-disc pl-5">
          {visibleEntries.map((info, index) => (
            <li key={`${info.model_name}-${info.deprecation_date}-${index}`}>{deprecationEntryText(info)}</li>
          ))}
          {remainingCount > 0 && <li>and {remainingCount} more</li>}
        </ul>
        <a href={uiHref("logging-and-alerts")} className="mt-2 inline-block font-medium underline">
          Get Slack alerts for deprecations
        </a>
      </AlertDescription>
    </Alert>
  );
};

export default ModelDeprecationBanner;
