import { useTranslation } from "react-i18next";
import { buttonVariants } from "@/components/ui/button";
import { useModelsInfo } from "@/app/(dashboard)/hooks/models/useModels";
import { modelGroupHref } from "@/utils/entityLinks";
import type { PromptCacheDetails } from "./alertPresentation";

const timeFormat = { dateStyle: "medium", timeStyle: "medium", timeZone: "UTC" } as const;
const SECONDS_PER_HOUR = 3600;

function useModelGroup(deploymentId: string): string | undefined {
  const info = useModelsInfo(1, 1, undefined, deploymentId);
  const name: unknown = info.data?.data[0]?.model_name;
  return typeof name === "string" && name ? name : undefined;
}

export default function PromptCacheEvidence({ details }: { details: PromptCacheDetails }) {
  const { t } = useTranslation();
  const deploymentId = details.deploymentId;
  const group = useModelGroup(deploymentId);
  const href = group ? modelGroupHref(group) : undefined;
  const ttl =
    details.ttlSeconds % SECONDS_PER_HOUR === 0
      ? t("alerts.unified.promptCache.ttlHours", { count: details.ttlSeconds / SECONDS_PER_HOUR })
      : t("alerts.unified.promptCache.ttlMinutes", { count: Math.round(details.ttlSeconds / 60) });
  const facts: [string, string][] = [
    [
      t("alerts.unified.promptCache.deployment"),
      group ? t("alerts.unified.promptCache.deploymentInGroup", { id: deploymentId, group }) : deploymentId,
    ],
    [t("alerts.unified.promptCache.model"), details.model],
    [t("alerts.unified.promptCache.misses"), details.misses.toLocaleString()],
    [
      t("alerts.unified.promptCache.provenWrite"),
      t("alerts.unified.promptCache.provenWriteTokens", { count: details.provenWriteTokens }),
    ],
    [t("alerts.unified.promptCache.ttl"), ttl],
    [
      t("alerts.unified.promptCache.window"),
      t("alerts.unified.promptCache.windowValue", {
        start: new Date(details.firstMiss).toLocaleString(undefined, timeFormat),
        end: new Date(details.lastMiss).toLocaleString(undefined, timeFormat),
      }),
    ],
  ];
  return (
    <section className="space-y-3" aria-labelledby="prompt-cache-evidence">
      <h2 id="prompt-cache-evidence" className="text-sm font-semibold">
        {t("alerts.unified.promptCache.evidenceHeading")}
      </h2>
      <dl className="grid gap-x-8 gap-y-3 text-sm sm:grid-cols-[max-content_minmax(0,1fr)]">
        {facts.map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="text-muted-foreground">{label}</dt>
            <dd className="break-all">{value}</dd>
          </div>
        ))}
      </dl>
      {href && (
        <a className={buttonVariants({ variant: "outline", size: "sm" })} href={href}>
          {t("alerts.unified.promptCache.modelLink")}
        </a>
      )}
      <p className="text-sm text-muted-foreground">{t("alerts.unified.promptCache.scope")}</p>
    </section>
  );
}
