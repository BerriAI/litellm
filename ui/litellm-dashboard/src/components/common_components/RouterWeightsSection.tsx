import { useQuery } from "@tanstack/react-query";
import { AlertCircle, X } from "lucide-react";
import { useId } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { modelInfoCall } from "../networking";
import {
  GroupWeights,
  RouterWeights,
  WeightableDeployment,
  hasNoPositiveWeight,
  parseWeightInput,
  weightRows,
  withDeploymentWeight,
  withGroup,
  withoutGroup,
} from "./routerWeights";

const WEIGHTED_STRATEGY = "simple-shuffle";
const DEPLOYMENTS_PAGE_SIZE = 1000;

interface RouterWeightsSectionProps {
  accessToken: string;
  teamId?: string | null;
  availableModels: string[];
  selectedStrategy: string | null;
  weights: RouterWeights;
  onChange: (weights: RouterWeights) => void;
}

interface WeightGroupProps {
  accessToken: string;
  teamId?: string | null;
  group: string;
  groupWeights: GroupWeights;
  onWeightChange: (deploymentId: string, weight: number | null) => void;
  onRemove: () => void;
}

function WeightGroup({ accessToken, teamId, group, groupWeights, onWeightChange, onRemove }: WeightGroupProps) {
  const { t } = useTranslation();
  const {
    data: deployments = [],
    isLoading,
    isError,
  } = useQuery<WeightableDeployment[]>({
    queryKey: ["routerWeightDeployments", accessToken, teamId ?? null, group],
    queryFn: async () =>
      (
        await modelInfoCall(
          accessToken,
          "",
          "",
          1,
          DEPLOYMENTS_PAGE_SIZE,
          undefined,
          undefined,
          teamId ?? undefined,
          undefined,
          undefined,
          undefined,
          group,
        )
      ).data ?? [],
    enabled: Boolean(accessToken),
  });
  const rows = weightRows(group, groupWeights, deployments, teamId ?? null);
  const noDeployments = !isLoading && !isError && rows.length === 0;

  return (
    <div className="rounded-md border border-border p-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-foreground">{group}</span>
        <Button
          type="button"
          variant="ghost"
          size="icon-xs"
          aria-label={t("routerWeights.removeGroup", { group })}
          onClick={onRemove}
        >
          <X />
        </Button>
      </div>
      {isError && (
        <p role="alert" className="text-xs text-destructive">
          {t("routerWeights.loadFailed")}
        </p>
      )}
      {noDeployments && <p className="text-xs text-muted-foreground">{t("routerWeights.noDeployments")}</p>}
      {rows.map((row) => (
        <div key={row.deploymentId} className="flex items-center gap-3">
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm text-foreground">
              {row.label ?? (isError ? "" : t("routerWeights.unknownDeployment"))}
            </div>
            <div className="truncate text-xs text-muted-foreground">{row.deploymentId}</div>
          </div>
          <Input
            type="number"
            min={0}
            step="any"
            className="w-28"
            aria-label={t("routerWeights.weightFor", {
              deployment: row.label ? `${row.label} (${row.deploymentId})` : row.deploymentId,
            })}
            value={row.weight ?? ""}
            onChange={(event) => {
              const weight = parseWeightInput(event.target.value);
              if (weight !== undefined) onWeightChange(row.deploymentId, weight);
            }}
          />
          <span className="w-24 text-right text-xs text-muted-foreground">
            {row.sharePercent === null ? "" : t("routerWeights.share", { percent: row.sharePercent })}
          </span>
        </div>
      ))}
      {hasNoPositiveWeight(groupWeights) && (
        <p role="alert" className="flex items-center gap-2 text-xs text-destructive">
          <AlertCircle className="size-4" />
          {t("routerWeights.noPositiveWeight")}
        </p>
      )}
    </div>
  );
}

export default function RouterWeightsSection({
  accessToken,
  teamId,
  availableModels,
  selectedStrategy,
  weights,
  onChange,
}: RouterWeightsSectionProps) {
  const { t } = useTranslation();
  const addGroupInputId = useId();
  const unweightedModels = availableModels.filter((model) => !(model in weights));
  const ignoredByStrategy = Boolean(selectedStrategy) && selectedStrategy !== WEIGHTED_STRATEGY;

  return (
    <div className="space-y-4">
      <div className="max-w-3xl">
        <h3 className="text-sm font-medium text-foreground">{t("routerWeights.title")}</h3>
        <p className="mt-1 text-xs text-muted-foreground">{t("routerWeights.description")}</p>
      </div>
      {ignoredByStrategy && Object.keys(weights).length > 0 && (
        <div className="flex items-center gap-2 rounded-sm bg-warning/10 p-2 text-xs text-warning">
          <AlertCircle className="size-4" />
          <span>{t("routerWeights.ignoredForStrategy", { strategy: selectedStrategy })}</span>
        </div>
      )}
      {Object.entries(weights).map(([group, groupWeights]) => (
        <WeightGroup
          key={group}
          accessToken={accessToken}
          teamId={teamId}
          group={group}
          groupWeights={groupWeights}
          onWeightChange={(deploymentId, weight) =>
            onChange(withDeploymentWeight(weights, group, deploymentId, weight))
          }
          onRemove={() => onChange(withoutGroup(weights, group))}
        />
      ))}
      <div className="max-w-md">
        <label htmlFor={addGroupInputId} className="mb-2 block text-sm font-medium text-foreground">
          {t("routerWeights.addGroup")}
        </label>
        <SearchSelect
          inputId={addGroupInputId}
          options={unweightedModels.map((model) => ({ label: model, value: model }))}
          value={null}
          onValueChange={(group) => {
            if (group) onChange(withGroup(weights, group));
          }}
          placeholder={t("routerWeights.addGroupPlaceholder")}
          emptyText={t("routerWeights.noModels")}
        />
      </div>
    </div>
  );
}
