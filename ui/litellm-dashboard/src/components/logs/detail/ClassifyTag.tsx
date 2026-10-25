"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/cva.config";

export const AUTOROUTER_CLASSIFIER_ORIGIN = "autorouter_classifier";

export function ClassifyTag({ origin, className }: { origin?: string | null; className?: string }) {
  const { t } = useLanguage();
  if (origin !== AUTOROUTER_CLASSIFIER_ORIGIN) return null;
  return (
    <Badge
      variant="secondary"
      title={t("Tier classification call made by the auto-router, not a request the caller sent")}
      className={cn("px-2 py-0 text-[10px] font-normal", className)}
    >
      {t("Classify")}
    </Badge>
  );
}
