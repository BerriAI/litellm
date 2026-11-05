import type { components } from "@/lib/http/schema";

type Recommendation = Pick<
  components["schemas"]["ModelDeprecationInfo"],
  "successor_model" | "successor_model_source" | "deprecation_source"
>;

const SOURCE_LABELS = {
  model_cost_map: "LiteLLM model cost map",
  deployment_config: "Deployment configuration",
} as const;

export default function RecommendedReplacement({ recommendation }: { recommendation: Recommendation }) {
  const { successor_model: model, successor_model_source: source, deprecation_source: notice } = recommendation;
  const sourceLabel = (source && SOURCE_LABELS[source]) || "Not provided";
  const noticeUrl = notice && /^https?:\/\//i.test(notice) ? notice : null;

  return (
    <div className="mt-2 space-y-1 text-xs text-muted-foreground">
      <p>
        Recommended new model:{" "}
        {model ? <code className="rounded bg-muted px-1.5 py-0.5 text-foreground">{model}</code> : "Not provided"}
      </p>
      {model && (
        <p className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span>
            Source:{" "}
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
              Provider deprecation notice ↗
            </a>
          )}
        </p>
      )}
    </div>
  );
}
