"use client";

import { ArrowUpRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import { DeprecationModel, DeprecationTiming } from "./DeprecationModel";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";
import { migrationHref } from "@/components/ModelDeprecationAlerts/migrationRoute";
import RecommendedReplacement from "./RecommendedReplacement";
import type { DeprecationAlert } from "./deprecationAlerts";

export default function DeprecationAlertItem({ alert }: { alert: DeprecationAlert }) {
  const { t } = useTranslation();
  const retired = alert.model.status === "deprecated";
  const title = retired
    ? t("alerts.alertItem.deprecatedTitle", { model: alert.usageModel })
    : t("alerts.alertItem.retiresTitle", { model: alert.usageModel, days: alert.model.days_until_deprecation });
  return (
    <article aria-label={title} className="flex flex-col gap-3">
      <div className="flex items-start gap-2.5">
        <div className="min-w-0 flex-1">
          <DeprecationModel alert={alert} />
          <p className="mt-1 text-xs">
            <DeprecationTiming alert={alert} />
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {t("alerts.alertItem.usage", { requests: alert.requests.toLocaleString(), date: alert.lastUsedDate })}
          </p>
          <RecommendedReplacement recommendation={alert.model} />
        </div>
      </div>
      <a className={cn(buttonVariants({ size: "sm" }), "self-start")} href={migrationHref(alert.usageModel)}>
        {t("alerts.alertItem.reviewMigration")} <ArrowUpRight className="size-3.5" />
      </a>
    </article>
  );
}
