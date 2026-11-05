"use client";

import { ArrowUpRight } from "lucide-react";
import { DeprecationModel, DeprecationTiming } from "./DeprecationModel";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";
import { migrationHref } from "@/components/ModelDeprecationAlerts/migrationRoute";
import RecommendedReplacement from "./RecommendedReplacement";
import type { DeprecationAlert } from "./deprecationAlerts";

export default function DeprecationAlertItem({ alert }: { alert: DeprecationAlert }) {
  const retired = alert.model.status === "deprecated";
  const title = retired
    ? `${alert.usageModel} is deprecated`
    : `${alert.usageModel} retires in ${alert.model.days_until_deprecation} days`;
  return (
    <article aria-label={title} className="flex flex-col gap-3">
      <div className="flex items-start gap-2.5">
        <div className="min-w-0 flex-1">
          <DeprecationModel alert={alert} />
          <p className="mt-1 text-xs">
            <DeprecationTiming alert={alert} />
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            {alert.requests.toLocaleString()} requests in the last 30 days · Last seen {alert.lastUsedDate} UTC
          </p>
          <RecommendedReplacement recommendation={alert.model} />
        </div>
      </div>
      <a className={cn(buttonVariants({ size: "sm" }), "self-start")} href={migrationHref(alert.usageModel)}>
        Review migration <ArrowUpRight className="size-3.5" />
      </a>
    </article>
  );
}
