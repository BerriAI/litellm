import type { AutoRouterRecommendationResponse } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import type { AutoSetupObjective, AutoSetupQualityLevel } from "./build_complexity_router_config";

const OBJECTIVE_DESCRIPTION: Record<AutoSetupObjective, string> = {
  cost: "Uses expected cost per completed benchmark task, including failures",
  task_completion_speed: "Uses live response speed for easy requests and benchmark completion time for hard ones",
  balanced: "Balances cost per completed task with those same completion-time signals",
};

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
  onUseManual: () => void;
}

const recommendationErrorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : "Could not build Auto setup";

const readyMessage = (recommendation: AutoRouterRecommendationResponse): string => {
  const count = recommendation.matched_model_groups.length;
  return `Ready with ${count} available model${count === 1 ? "" : "s"}. Review or edit below`;
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
  onUseManual,
}: AutoSetupControlsProps) {
  return (
    <div className="space-y-4 rounded-lg border border-border p-4" data-testid="auto-setup-controls">
      <div>
        <div className="font-medium text-foreground">Auto setup</div>
        <div className="text-xs text-muted-foreground">
          Builds an editable complexity router from the models available to you
        </div>
      </div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div>
          <label className="mb-2 block text-sm font-medium text-foreground">Quality level</label>
          <Select
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
        <div className="space-y-1">
          <div className={`text-xs ${hasAutoPolicy ? "text-success" : "text-warning"}`}>
            {hasAutoPolicy ? readyMessage(recommendation) : "Customized model pools now use standard manual selection"}
          </div>
          {!hasAutoPolicy && (
            <button type="button" className="text-xs text-primary underline" onClick={onRegenerate}>
              Regenerate Auto setup
            </button>
          )}
        </div>
      )}
      <Button type="button" variant="ghost" className="h-auto px-0" onClick={onUseManual}>
        Choose a template or set up manually
      </Button>
    </div>
  );
}
