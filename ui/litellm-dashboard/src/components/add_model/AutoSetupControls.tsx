import type { AutoRouterRecommendationResponse } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import type { AutoSetupObjective, AutoSetupQualityLevel } from "./build_complexity_router_config";

const OBJECTIVE_DESCRIPTION: Record<AutoSetupObjective, string> = {
  cost: "Spend less while staying within your selected quality level",
  task_completion_speed: "Finish requests sooner, not just generate tokens faster",
  balanced: "Balance expected cost with time to completion",
};

const QUALITY_OPTIONS = [
  { value: "economy", label: "Economy" },
  { value: "balanced", label: "Balanced" },
  { value: "high", label: "High" },
  { value: "max", label: "Max" },
];

const OBJECTIVE_OPTIONS = [
  { value: "cost", label: "Cost" },
  { value: "task_completion_speed", label: "Task completion speed" },
  { value: "balanced", label: "Balanced" },
];

interface AutoSetupControlsProps {
  qualityLevel: AutoSetupQualityLevel;
  optimizeFor: AutoSetupObjective;
  onQualityLevelChange: (value: AutoSetupQualityLevel) => void;
  onOptimizeForChange: (value: AutoSetupObjective) => void;
  recommendation?: AutoRouterRecommendationResponse;
  recommendationEnabled: boolean;
  recommendationPending: boolean;
  recommendationFailed: boolean;
  recommendationError: unknown;
  requiresTeamScope: boolean;
  hasAutoPolicy: boolean;
  onRetry: () => void;
  onRegenerate: () => void;
  onBack: () => void;
}

const recommendationErrorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : "Could not build Auto setup";

const readyMessage = (recommendation: AutoRouterRecommendationResponse): string => {
  const count = recommendation.matched_model_groups.length;
  const total = recommendation.available_model_group_count;
  return `Uses ${count} of ${total} available model${total === 1 ? "" : "s"} across four complexity levels`;
};

const recommendationDescription = (optimizeFor: AutoSetupObjective): string => {
  switch (optimizeFor) {
    case "cost":
      return "Favors the lowest expected cost while keeping each request within your selected quality level";
    case "task_completion_speed":
      return "Easy requests use live response speed; difficult requests use benchmark completion time";
    case "balanced":
      return "Balances expected cost with response speed and benchmark completion time";
  }
};

export default function AutoSetupControls({
  qualityLevel,
  optimizeFor,
  onQualityLevelChange,
  onOptimizeForChange,
  recommendation,
  recommendationEnabled,
  recommendationPending,
  recommendationFailed,
  recommendationError,
  requiresTeamScope,
  hasAutoPolicy,
  onRetry,
  onRegenerate,
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
          {hasAutoPolicy && excludedCount > 0 && (
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
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
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
            Higher levels keep each tier closer to the smartest available model
          </div>
        </div>
        <div>
          <label className="mb-2 block text-sm font-medium text-foreground">Optimize for</label>
          <Select
            items={OBJECTIVE_OPTIONS}
            value={optimizeFor}
            onValueChange={(value: AutoSetupObjective | null) => value && onOptimizeForChange(value)}
          >
            <SelectTrigger data-testid="auto-objective-selector" className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="cost">Cost</SelectItem>
              <SelectItem value="task_completion_speed">Task completion speed</SelectItem>
              <SelectItem value="balanced">Balanced</SelectItem>
            </SelectContent>
          </Select>
          <div className="mt-1 text-xs text-muted-foreground">{OBJECTIVE_DESCRIPTION[optimizeFor]}</div>
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
          <div className={`text-sm font-medium ${hasAutoPolicy ? "text-foreground" : "text-warning"}`}>
            {hasAutoPolicy ? "Recommended setup" : "Configuration edited"}
          </div>
          <div className="text-xs text-muted-foreground">
            {hasAutoPolicy
              ? `${readyMessage(recommendation)}. ${recommendationDescription(optimizeFor)}`
              : "Your model changes are now treated as a manual configuration"}
          </div>
          {!hasAutoPolicy && (
            <button type="button" className="text-xs text-primary underline" onClick={onRegenerate}>
              Restore recommended setup
            </button>
          )}
        </div>
      )}
    </div>
  );
}
