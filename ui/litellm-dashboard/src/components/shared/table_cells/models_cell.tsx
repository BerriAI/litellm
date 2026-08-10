"use client";

import { deriveKeyModelScope } from "@/components/key_scope";
import { getModelDisplayName } from "@/components/key_team_helpers/fetch_available_models_team_key";
import { Badge } from "@/components/ui/badge";

import { CellTooltip } from "./cell_tooltip";

interface ModelsCellProps {
  models: string[] | null | undefined;
  maxVisible?: number;
  allowedRoutes?: string[] | null;
  keyType?: string | null;
  labels?: {
    allProxyModels: string;
    noModelAccess: string;
    scopedRoutes: (scope: string) => string;
    more: (count: number) => string;
  };
}

const WILDCARD_MODEL = "all-proxy-models";

const formatModel = (model: string, allProxyModelsLabel: string): string => {
  if (model === WILDCARD_MODEL) {
    return allProxyModelsLabel;
  }
  const name = getModelDisplayName(model);
  return name.length > 30 ? `${name.slice(0, 30)}...` : name;
};

const DEFAULT_LABELS = {
  allProxyModels: "All Proxy Models",
  noModelAccess: "No model access",
  scopedRoutes: (scope: string) => `Scoped to ${scope} routes; this key cannot call any models`,
  more: (count: number) => `+${count} more`,
};

export function ModelsCell({
  models,
  maxVisible = 3,
  allowedRoutes,
  keyType,
  labels = DEFAULT_LABELS,
}: ModelsCellProps) {
  if (!Array.isArray(models) || models.length === 0) {
    const scope = deriveKeyModelScope(allowedRoutes, keyType);
    if (!scope.hasModelAccess) {
      return (
        <CellTooltip
          content={labels.scopedRoutes(scope.label)}
          trigger={
            <Badge variant="secondary" className="cursor-default">
              {labels.noModelAccess}
            </Badge>
          }
        />
      );
    }
    return <Badge variant="secondary">{labels.allProxyModels}</Badge>;
  }

  const visible = models.slice(0, maxVisible);
  const overflow = models.slice(maxVisible);

  return (
    <div className="flex flex-wrap items-center gap-1">
      {visible.map((model, index) => (
        <Badge key={index} variant={model === WILDCARD_MODEL ? "secondary" : "outline"}>
          {formatModel(model, labels.allProxyModels)}
        </Badge>
      ))}
      {overflow.length > 0 && (
        <CellTooltip
          content={
            <div className="flex max-w-[280px] flex-col gap-0.5">
              {overflow.map((model, index) => (
                <span key={index}>{formatModel(model, labels.allProxyModels)}</span>
              ))}
            </div>
          }
          trigger={
            <Badge variant="outline" className="cursor-default">
              {labels.more(overflow.length)}
            </Badge>
          }
        />
      )}
    </div>
  );
}
