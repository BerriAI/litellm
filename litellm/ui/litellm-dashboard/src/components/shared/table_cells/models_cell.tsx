"use client";

import { useTranslation } from "react-i18next";

import { deriveKeyModelScope } from "@/components/key_scope";
import { getModelDisplayName } from "@/components/key_team_helpers/fetch_available_models_team_key";
import { Badge } from "@/components/ui/badge";
import i18n from "@/i18n/i18n";

import { CellTooltip } from "./cell_tooltip";

interface ModelsCellProps {
  models: string[] | null | undefined;
  maxVisible?: number;
  allowedRoutes?: string[] | null;
  keyType?: string | null;
}

const WILDCARD_MODEL = "all-proxy-models";

const formatModel = (model: string): string => {
  if (model === WILDCARD_MODEL) {
    return i18n.t("shared.tableCells.modelsCell.allProxyModels");
  }
  const name = getModelDisplayName(model);
  return name.length > 30 ? `${name.slice(0, 30)}...` : name;
};

export function ModelsCell({ models, maxVisible = 3, allowedRoutes, keyType }: ModelsCellProps) {
  const { t } = useTranslation();
  if (!Array.isArray(models) || models.length === 0) {
    const scope = deriveKeyModelScope(allowedRoutes, keyType);
    if (!scope.hasModelAccess) {
      return (
        <CellTooltip
          content={t("shared.tableCells.modelsCell.scopedTooltip", { scope: scope.label })}
          trigger={
            <Badge variant="secondary" className="cursor-default">
              {t("shared.tableCells.modelsCell.noModelAccess")}
            </Badge>
          }
        />
      );
    }
    return <Badge variant="secondary">{t("shared.tableCells.modelsCell.allProxyModels")}</Badge>;
  }

  const visible = models.slice(0, maxVisible);
  const overflow = models.slice(maxVisible);

  return (
    <div className="flex flex-wrap items-center gap-1">
      {visible.map((model, index) => (
        <Badge key={index} variant={model === WILDCARD_MODEL ? "secondary" : "outline"}>
          {formatModel(model)}
        </Badge>
      ))}
      {overflow.length > 0 && (
        <CellTooltip
          content={
            <div className="flex max-w-[280px] flex-col gap-0.5">
              {overflow.map((model, index) => (
                <span key={index}>{formatModel(model)}</span>
              ))}
            </div>
          }
          trigger={
            <Badge variant="outline" className="cursor-default">
              {t("shared.tableCells.modelsCell.more", { number: overflow.length })}
            </Badge>
          }
        />
      )}
    </div>
  );
}
