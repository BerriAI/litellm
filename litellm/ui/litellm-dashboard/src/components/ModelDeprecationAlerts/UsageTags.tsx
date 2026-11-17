"use client";

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/badge";

export default function UsageTags({
  tags,
  loading,
  failed,
}: {
  tags: readonly string[];
  loading: boolean;
  failed: boolean;
}) {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(false);
  if (loading) return <span className="text-xs text-muted-foreground">{t("alerts.usageTags.loading")}</span>;
  if (failed) return <span className="text-xs text-muted-foreground">{t("alerts.usageTags.failed")}</span>;
  if (tags.length === 0) return <span className="text-xs text-muted-foreground">{t("alerts.usageTags.none")}</span>;
  const visible = expanded ? tags : tags.slice(0, 5);
  return (
    <div className="flex max-w-72 flex-wrap items-center gap-1">
      {visible.map((tag) => (
        <Badge key={tag} variant="secondary" className="max-w-full whitespace-normal break-all font-normal">
          {tag}
        </Badge>
      ))}
      {tags.length > 5 && (
        <button
          type="button"
          className="text-xs text-primary hover:underline"
          aria-expanded={expanded}
          aria-label={
            expanded ? t("alerts.usageTags.showFewer") : t("alerts.usageTags.showMore", { hidden: tags.length - 5 })
          }
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? t("alerts.usageTags.showLess") : t("alerts.usageTags.more", { hidden: tags.length - 5 })}
        </button>
      )}
    </div>
  );
}
