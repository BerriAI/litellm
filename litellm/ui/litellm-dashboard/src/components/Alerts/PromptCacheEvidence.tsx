import { useTranslation } from "react-i18next";
import type { PromptCacheDetails } from "./alertPresentation";

const timeFormat = { dateStyle: "medium", timeStyle: "medium", timeZone: "UTC" } as const;

export default function PromptCacheEvidence({ details }: { details: PromptCacheDetails }) {
  const { t } = useTranslation();
  const facts: [string, string][] = [
    [t("alerts.unified.promptCache.deployment"), details.deploymentId],
    [t("alerts.unified.promptCache.model"), details.model],
    [t("alerts.unified.promptCache.misses"), details.misses.toLocaleString()],
    [
      t("alerts.unified.promptCache.provenWrite"),
      t("alerts.unified.promptCache.provenWriteTokens", { count: details.provenWriteTokens }),
    ],
    [
      t("alerts.unified.promptCache.ttl"),
      t("alerts.unified.promptCache.ttlMinutes", { count: details.ttlSeconds / 60 }),
    ],
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
      <p className="text-sm text-muted-foreground">{t("alerts.unified.promptCache.scope")}</p>
    </section>
  );
}
