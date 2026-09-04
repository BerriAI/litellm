import type { AutoRouterRecommendationResponse } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import type { AutoSetupQualityLevel } from "./build_complexity_router_config";

const QUALITY_OPTIONS = [
  { value: "economy", label: "Economy" },
  { value: "balanced", label: "Balanced" },
  { value: "high", label: "High" },
  { value: "max", label: "Max" },
];

const QUALITY_DESCRIPTION: Record<AutoSetupQualityLevel, string> = {
  economy: "Allows models up to 15 points below the smartest available option",
  balanced: "Keeps models within 7 points of the smartest available option",
  high: "Keeps models within 3 points of the smartest available option",
  max: "Keeps models within 1 point of the smartest available option",
};

interface AutoSetupControlsProps {
  qualityLevel: AutoSetupQualityLevel;
  onQualityLevelChange: (value: AutoSetupQualityLevel) => void;
  recommendation?: AutoRouterRecommendationResponse;
  recommendationEnabled: boolean;
  recommendationPending: boolean;
  recommendationFailed: boolean;
  recommendationError: unknown;
  requiresTeamScope: boolean;
  onRetry: () => void;
  onBack: () => void;
}

const recommendationErrorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : "Could not build Auto setup";

const readyMessage = (recommendation: AutoRouterRecommendationResponse): string => {
  const count = recommendation.matched_model_groups.length;
  const total = recommendation.available_model_group_count;
  return `Uses ${count} of ${total} available model${total === 1 ? "" : "s"} across four complexity levels`;
};

export default function AutoSetupControls({
  qualityLevel,
  onQualityLevelChange,
  recommendation,
  recommendationEnabled,
  recommendationPending,
  recommendationFailed,
  recommendationError,
  requiresTeamScope,
  onRetry,
  onBack,
}: AutoSetupControlsProps) {
  const excludedCount = recommendation?.excluded_model_groups.length ?? 0;

  return (
    <div className="space-y-4" data-testid="auto-setup-controls">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="font-medium text-foreground">Configure automatically</div>
          <div className="text-xs text-muted-foreground">
            Choose the outcome you want. LiteLLM will build the model tiers for you
          </div>
          {recommendation && excludedCount > 0 && (
            <div className="text-xs text-muted-foreground">
              {excludedCount} model{excludedCount === 1 ? " was" : "s were"} left out because Auto setup could not
              compare {excludedCount === 1 ? "it" : "them"} safely
            </div>
          )}
        </div>
        <Button type="button" variant="ghost" className="h-auto shrink-0 px-0" onClick={onBack}>
          Back
        </Button>
      </div>
      <div>
        <div>
          <label className="mb-2 block text-sm font-medium text-foreground">Quality level</label>
          <Select
            items={QUALITY_OPTIONS}
            value={qualityLevel}
            onValueChange={(value: AutoSetupQualityLevel | null) => value && onQualityLevelChange(value)}
          >
            <SelectTrigger data-testid="auto-quality-selector" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="economy">Economy</SelectItem>
              <SelectItem value="balanced">Balanced</SelectItem>
              <SelectItem value="high">High</SelectItem>
              <SelectItem value="max">Max</SelectItem>
            </SelectContent>
          </Select>
          <div className="mt-1 text-xs text-muted-foreground">
            {QUALITY_DESCRIPTION[qualityLevel]}. Auto picks the lowest estimated completion cost inside that range, or
            the smartest option when price is unavailable
          </div>
          <div className="mt-1 text-xs text-muted-foreground">
            Quality uses uncertainty-adjusted public benchmark scores for each complexity level
          </div>
        </div>
      </div>
      {recommendationPending && recommendation === undefined && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <UiLoadingSpinner className="size-3" /> Building your router...
        </div>
      )}
      {!recommendationEnabled && requiresTeamScope && (
        <div className="text-xs text-muted-foreground">Select a team to build its Auto setup</div>
      )}
      {recommendationFailed && recommendation === undefined && (
        <div className="text-xs text-destructive">
          {recommendationErrorMessage(recommendationError)}.{" "}
          <button type="button" className="underline" onClick={onRetry}>
            Retry
          </button>
        </div>
      )}
      {recommendation && (
        <div className="space-y-2 rounded-lg border border-border bg-muted/40 p-3" data-testid="auto-setup-summary">
          <div className="text-sm font-medium text-foreground">Recommended setup</div>
          <div className="text-xs text-muted-foreground">
            {readyMessage(recommendation)}. Each tier is the lowest-cost model with comparable pricing that meets your
            selected quality level
          </div>
        </div>
      )}
    </div>
  );
}
