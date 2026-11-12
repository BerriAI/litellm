import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";
import type { components } from "@/lib/http/schema";

type Recommendation = Pick<
  components["schemas"]["ModelDeprecationInfo"],
  "successor_model" | "successor_model_source" | "deprecation_source"
>;

const SOURCE_LABEL_KEYS = {
  model_cost_map: "alerts.recommendation.sources.modelCostMap",
  deployment_config: "alerts.recommendation.sources.deploymentConfig",
} as const satisfies Record<string, ParseKeys>;

export default function RecommendedReplacement({ recommendation }: { recommendation: Recommendation }) {
  const { t } = useTranslation();
  const { successor_model: model, successor_model_source: source, deprecation_source: notice } = recommendation;
  const sourceLabel =
    source && Object.hasOwn(SOURCE_LABEL_KEYS, source)
      ? t(SOURCE_LABEL_KEYS[source])
      : t("alerts.recommendation.notProvided");
  const noticeUrl = notice && /^https?:\/\//i.test(notice) ? notice : null;

  return (
    <div className="mt-2 space-y-1 text-xs text-muted-foreground">
      <p>
        {t("alerts.recommendation.recommendedModel")}{" "}
        {model ? (
          <code className="rounded bg-muted px-1.5 py-0.5 text-foreground">{model}</code>
        ) : (
          t("alerts.recommendation.notProvided")
        )}
      </p>
      {model && (
        <p className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span>
            {t("alerts.recommendation.source")}{" "}
            {source === "model_cost_map" ? (
              <a
                href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
                target="_blank"
                rel="noopener noreferrer"
                className="underline underline-offset-2"
              >
                {sourceLabel}
              </a>
            ) : (
              sourceLabel
            )}
          </span>
          {noticeUrl && (
            <a href={noticeUrl} target="_blank" rel="noopener noreferrer" className="underline underline-offset-2">
              {t("alerts.recommendation.providerNotice")}
            </a>
          )}
        </p>
      )}
    </div>
  );
}
