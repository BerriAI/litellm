"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import { ColumnDef } from "@tanstack/react-table";
import { englishTranslate, type Language, type Translate } from "@/i18n/language";
import { CredentialIdentity } from "@/components/shared/CredentialIdentity";
import { Copy, Info, Loader2, Pencil, RefreshCw, Trash2 } from "lucide-react";

import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { ModelData } from "@/components/model_dashboard/types";
import { DataTableSortHeader } from "@/components/shared/DataTable";
import { CellTooltip, DateCell, formatCellDate, IdCell, StatusBadge } from "@/components/shared/table_cells";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Switch } from "@/components/ui/switch";
import { getDisplayModelName } from "@/components/view_model/model_name_display";
import { copyToClipboard, formatPerSecondCost } from "@/utils/dataUtils";

export const MODEL_ID_COLUMN_ID = "model_info_id";
export const MODEL_NAME_COLUMN_ID = "model_name";
export const CREDENTIALS_COLUMN_ID = "litellm_credential_name";
export const CREATED_BY_COLUMN_ID = "model_info_created_by";
export const UPDATED_AT_COLUMN_ID = "model_info_updated_at";
export const COSTS_COLUMN_ID = "input_cost";
export const TEAM_ID_COLUMN_ID = "model_info_team_id";
export const ACCESS_GROUPS_COLUMN_ID = "model_info_access_groups";
export const STATUS_COLUMN_ID = "model_info_db_model";

export const MODEL_TABLE_SORT_COLUMN_IDS = [
  MODEL_NAME_COLUMN_ID,
  CREATED_BY_COLUMN_ID,
  UPDATED_AT_COLUMN_ID,
  COSTS_COLUMN_ID,
  STATUS_COLUMN_ID,
] as const;

export type ModelTableSortColumnId = (typeof MODEL_TABLE_SORT_COLUMN_IDS)[number];

export const isModelTableSortColumnId = (columnId: string): columnId is ModelTableSortColumnId =>
  (MODEL_TABLE_SORT_COLUMN_IDS as readonly string[]).includes(columnId);

const COLUMN_ID_TO_SERVER_SORT_FIELD: Record<string, string> = {
  [COSTS_COLUMN_ID]: "costs",
  [STATUS_COLUMN_ID]: "status",
  [CREATED_BY_COLUMN_ID]: "created_at",
  [UPDATED_AT_COLUMN_ID]: "updated_at",
};

export const toServerSortField = (columnId: string): string => COLUMN_ID_TO_SERVER_SORT_FIELD[columnId] ?? columnId;

const formatShortDate = (value: string | null | undefined, language: Language): string | null => {
  if (!value) {
    return null;
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : formatCellDate(date, "date", language);
};

function ModelInformationCell({ model, displayName }: { model: ModelData; displayName: string }) {
  const { t } = useLanguage();
  const litellmModelName = model.litellm_model_name || "-";

  return (
    <HoverCard>
      <HoverCardTrigger
        render={
          <div className="flex min-w-0 items-center gap-2.5" data-testid={`model-information-${model.model_info.id}`} />
        }
      >
        {model.provider ? (
          <ProviderLogo provider={model.provider} className="size-6 shrink-0" />
        ) : (
          <span className="flex size-6 shrink-0 items-center justify-center rounded-md bg-muted text-xs text-muted-foreground">
            -
          </span>
        )}
        <span className="flex min-w-0 flex-col gap-0.5">
          <span className="max-w-60 truncate text-sm font-medium text-foreground" title={displayName}>
            {displayName}
          </span>
          <span className="max-w-60 truncate font-mono text-xs text-muted-foreground" title={litellmModelName}>
            {litellmModelName}
          </span>
        </span>
      </HoverCardTrigger>
      <HoverCardContent align="start" className="w-80">
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2">
            {model.provider ? <ProviderLogo provider={model.provider} className="size-4 shrink-0" /> : null}
            <span className="truncate text-xs text-muted-foreground">{model.provider || t("Unknown provider")}</span>
          </div>
          <div className="flex flex-col gap-0.5">
            <span className="text-xs text-muted-foreground">{t("Public Model Name")}</span>
            <span className="truncate text-sm font-medium text-foreground" title={displayName}>
              {displayName}
            </span>
          </div>
          <div className="flex flex-col gap-0.5">
            <span className="text-xs text-muted-foreground">{t("LiteLLM Model Name")}</span>
            <span className="flex min-w-0 items-center gap-1.5">
              <span className="truncate font-mono text-sm text-foreground" title={litellmModelName}>
                {litellmModelName}
              </span>
              <button
                type="button"
                aria-label={t("Copy LiteLLM model name")}
                data-testid={`copy-litellm-model-name-${model.model_info.id}`}
                className="shrink-0 cursor-pointer text-muted-foreground hover:text-foreground"
                onClick={() => void copyToClipboard(litellmModelName, t("LiteLLM model name copied"))}
              >
                <Copy className="size-3.5" />
              </button>
            </span>
          </div>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}

function CredentialsHeader() {
  const { t } = useLanguage();
  return (
    <span className="flex items-center gap-1">
      {t("Credentials")}
      <HoverCard>
        <HoverCardTrigger
          render={
            <button
              type="button"
              aria-label={t("About credential types")}
              data-testid="credentials-header-info"
              className="cursor-pointer text-muted-foreground hover:text-foreground"
            />
          }
        >
          <Info className="size-3.5" />
        </HoverCardTrigger>
        <HoverCardContent align="start" className="w-80">
          <div className="flex flex-col gap-3">
            <span className="text-sm font-medium text-foreground">{t("Credential types")}</span>
            <div className="flex flex-col gap-1">
              <span className="flex items-center gap-1.5 text-sm font-medium text-info">
                <RefreshCw className="size-3.5" />
                {t("Reusable")}
              </span>
              <span className="text-xs text-muted-foreground">
                {t("Credentials saved in LiteLLM that can be added to models repeatedly.")}
              </span>
            </div>
            <div className="flex flex-col gap-1">
              <span className="flex items-center gap-1.5 text-sm font-medium text-foreground">
                <Pencil className="size-3.5" />
                {t("Manual")}
              </span>
              <span className="text-xs text-muted-foreground">
                {t("Credentials added directly during model creation or defined in the config file.")}
              </span>
            </div>
          </div>
        </HoverCardContent>
      </HoverCard>
    </span>
  );
}

function CredentialsCell({ credentialName, label }: { credentialName: string | undefined; label: string | undefined }) {
  const { t } = useLanguage();
  if (!credentialName) {
    return (
      <Badge variant="outline" className="gap-1 font-normal text-muted-foreground">
        <Pencil className="size-3" />
        {t("Manual")}
      </Badge>
    );
  }

  return (
    <CredentialIdentity
      credentialName={credentialName}
      label={label}
      icon={<RefreshCw className="size-3 shrink-0" />}
      className="text-xs font-medium text-info"
    />
  );
}

function CreatedByCell({ model }: { model: ModelData }) {
  const { t, language } = useLanguage();
  const isConfigModel = !model.model_info?.db_model;
  const createdAt = formatShortDate(model.model_info.created_at, language);
  // 仅翻译本地占位，不翻译创建人等用户数据。
  const primary = isConfigModel ? t("Defined in config") : model.model_info.created_by || t("Unknown");
  const secondaryForDbModel = createdAt ?? t("Unknown date");

  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <span className="max-w-44 truncate text-sm text-foreground" title={primary}>
        {primary}
      </span>
      <span className="truncate text-xs text-muted-foreground">{isConfigModel ? "-" : secondaryForDbModel}</span>
    </div>
  );
}

function CostRow({ label, value }: { label: string; value: string }) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="text-[10px] font-semibold tracking-wider text-muted-foreground">{label}</span>
      <span className="text-xs font-medium tabular-nums text-foreground">{value}</span>
    </span>
  );
}

function CostsCell({ model }: { model: ModelData }) {
  const { t } = useLanguage();
  const { input_cost: inputCost, output_cost: outputCost, output_cost_per_second: perSecond } = model;
  const hasPerSecond = perSecond != null;
  const showInput = inputCost != null && (!hasPerSecond || Number(inputCost) > 0);
  const showOutput = outputCost != null && (!hasPerSecond || Number(outputCost) > 0);

  if (!showInput && !showOutput && !hasPerSecond) {
    return <span className="text-sm text-muted-foreground">-</span>;
  }

  return (
    <CellTooltip
      content={hasPerSecond ? t("Cost per 1M tokens; /s is cost per second of output") : t("Cost per 1M tokens")}
      trigger={
        <div className="flex flex-col gap-0.5 whitespace-nowrap">
          {showInput && <CostRow label={t("IN")} value={`$${inputCost}`} />}
          {showOutput && <CostRow label={t("OUT")} value={`$${outputCost}`} />}
          {hasPerSecond && <CostRow label={t("OUT")} value={formatPerSecondCost(perSecond)} />}
        </div>
      }
    />
  );
}

function AccessGroupsCell({ accessGroups }: { accessGroups: string[] | null }) {
  const { t } = useLanguage();
  if (!accessGroups || accessGroups.length === 0) {
    return <span className="text-sm text-muted-foreground">-</span>;
  }

  const [first, ...overflow] = accessGroups;

  return (
    <div className="flex min-w-0 items-center gap-1">
      <Badge variant="outline" className="max-w-36 truncate border-info/20 bg-info/10 font-normal text-info">
        {first}
      </Badge>
      {overflow.length > 0 && (
        <CellTooltip
          content={
            <div className="flex max-w-[280px] flex-col gap-0.5">
              {overflow.map((group) => (
                <span key={group}>{group}</span>
              ))}
            </div>
          }
          trigger={
            <Badge variant="outline" className="shrink-0 cursor-default font-normal">
              +{overflow.length} {t("more")}
            </Badge>
          }
        />
      )}
    </div>
  );
}

interface ModelRowActionsProps {
  model: ModelData;
  userRole: string;
  userID: string;
  isViewOnly: boolean;
  isPausing: boolean;
  onDeleteClick?: (modelId: string) => void;
  onTogglePauseClick?: (modelId: string, blocked: boolean) => void | Promise<void>;
}

function ModelRowActions({
  model,
  userRole,
  userID,
  isViewOnly,
  isPausing,
  onDeleteClick,
  onTogglePauseClick,
}: ModelRowActionsProps) {
  const { t } = useLanguage();
  const modelId = model.model_info?.id;
  const isConfigModel = !model.model_info?.db_model;
  const isAdmin = userRole === "Admin" && !isViewOnly;
  const canEditModel = !isViewOnly && (isAdmin || model.model_info?.created_by === userID);
  const isBlocked = model.model_info?.blocked === true;
  const isPauseToggleable = !isConfigModel && isAdmin && Boolean(onTogglePauseClick);

  const resolvePauseTooltip = (): string => {
    if (isConfigModel) {
      return t("Config models cannot be paused from the dashboard. Pause is DB-backed.");
    }
    if (!isAdmin) {
      return t("Only proxy admins can pause or resume a model.");
    }
    return isBlocked
      ? t("Resume model — restore normal routing.")
      : t("Pause model — stop routing requests until resumed.");
  };

  const deleteTooltip = isConfigModel
    ? t("Config model cannot be deleted on the dashboard. Please delete it from the config file.")
    : t("Delete model");

  return (
    <div className="flex items-center justify-end gap-1.5">
      <span className="flex w-8 shrink-0 items-center justify-center">
        {isPausing ? (
          <Loader2
            className="size-4 animate-spin text-muted-foreground"
            data-testid={`model-pause-pending-${modelId}`}
          />
        ) : (
          <CellTooltip
            content={resolvePauseTooltip()}
            trigger={
              <span className="inline-flex">
                <Switch
                  size="sm"
                  checked={!isBlocked}
                  disabled={!isPauseToggleable}
                  aria-label={isBlocked ? t("Resume model") : t("Pause model")}
                  data-testid={`model-pause-toggle-${modelId}`}
                  onCheckedChange={(nextChecked) => {
                    if (isPauseToggleable && onTogglePauseClick && modelId) {
                      void onTogglePauseClick(modelId, !nextChecked);
                    }
                  }}
                />
              </span>
            }
          />
        )}
      </span>
      <CellTooltip
        content={deleteTooltip}
        trigger={
          <span className="inline-flex">
            <Button
              variant="ghost"
              size="icon-sm"
              aria-label={t("Delete model")}
              data-testid={`model-delete-${modelId}`}
              disabled={isConfigModel || !canEditModel}
              className="text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
              onClick={() => {
                if (onDeleteClick && modelId) {
                  onDeleteClick(modelId);
                }
              }}
            >
              <Trash2 className="size-4" />
            </Button>
          </span>
        }
      />
    </div>
  );
}

export interface ModelsTableColumnDeps {
  t?: Translate;
  userRole: string;
  userID: string;
  isViewOnly: boolean;
  onModelIdClick: (modelId: string) => void;
  onTeamIdClick: (teamId: string) => void;
  onDeleteClick?: (modelId: string) => void;
  onTogglePauseClick?: (modelId: string, blocked: boolean) => void | Promise<void>;
  pausingModelId?: string | null;
  credentialLabels?: ReadonlyMap<string, string>;
}

export const getModelsTableColumns = ({
  t = englishTranslate,
  userRole,
  userID,
  isViewOnly,
  onModelIdClick,
  onTeamIdClick,
  onDeleteClick,
  onTogglePauseClick,
  pausingModelId,
  credentialLabels,
}: ModelsTableColumnDeps): ColumnDef<ModelData>[] => [
  {
    id: MODEL_ID_COLUMN_ID,
    accessorFn: (row) => row.model_info.id,
    meta: { title: t("Model ID") },
    header: t("Model ID"),
    enableSorting: false,
    size: 140,
    minSize: 90,
    cell: ({ row }) => (
      <IdCell
        value={row.original.model_info.id}
        onClick={onModelIdClick}
        dataTestId={`model-id-${row.original.model_info.id}`}
      />
    ),
  },
  {
    id: MODEL_NAME_COLUMN_ID,
    accessorFn: (row) => row.model_name ?? "",
    meta: { title: t("Model Information"), skeleton: "twoLine" },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Model Information")} />,
    enableSorting: true,
    size: 280,
    minSize: 160,
    cell: ({ row }) => (
      <ModelInformationCell model={row.original} displayName={getDisplayModelName(row.original) || "-"} />
    ),
  },
  {
    id: CREDENTIALS_COLUMN_ID,
    accessorFn: (row) => row.litellm_params?.litellm_credential_name ?? "",
    meta: { title: t("Credentials") },
    header: () => <CredentialsHeader />,
    enableSorting: false,
    size: 180,
    minSize: 110,
    cell: ({ row }) => {
      const credentialName = row.original.litellm_params?.litellm_credential_name;
      return (
        <CredentialsCell
          credentialName={credentialName}
          label={credentialName ? credentialLabels?.get(credentialName) : undefined}
        />
      );
    },
  },
  {
    id: CREATED_BY_COLUMN_ID,
    accessorFn: (row) => row.model_info.created_by ?? "",
    meta: { title: t("Created By"), skeleton: "twoLine" },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Created By")} />,
    enableSorting: true,
    size: 180,
    minSize: 110,
    cell: ({ row }) => <CreatedByCell model={row.original} />,
  },
  {
    id: UPDATED_AT_COLUMN_ID,
    accessorFn: (row) => row.model_info.updated_at ?? "",
    meta: { title: t("Updated At") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Updated At")} />,
    enableSorting: true,
    size: 140,
    minSize: 100,
    cell: ({ row }) => <DateCell value={row.original.model_info.updated_at} precision="date" />,
  },
  {
    id: COSTS_COLUMN_ID,
    accessorFn: (row) => row.input_cost,
    meta: { title: t("Costs"), numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Costs")} />,
    enableSorting: true,
    size: 130,
    minSize: 90,
    cell: ({ row }) => <CostsCell model={row.original} />,
  },
  {
    id: TEAM_ID_COLUMN_ID,
    accessorFn: (row) => row.model_info.team_id ?? "",
    meta: { title: t("Team ID") },
    header: t("Team ID"),
    enableSorting: false,
    size: 140,
    minSize: 90,
    cell: ({ row }) => (
      <IdCell
        value={row.original.model_info.team_id}
        onClick={onTeamIdClick}
        dataTestId={`model-team-id-${row.original.model_info.id}`}
      />
    ),
  },
  {
    id: ACCESS_GROUPS_COLUMN_ID,
    accessorFn: (row) => row.model_info.access_groups ?? [],
    meta: { title: t("Model Access Group"), skeleton: "chips" },
    header: t("Model Access Group"),
    enableSorting: false,
    size: 200,
    minSize: 120,
    cell: ({ row }) => <AccessGroupsCell accessGroups={row.original.model_info.access_groups} />,
  },
  {
    id: STATUS_COLUMN_ID,
    accessorFn: (row) => row.model_info.db_model,
    meta: { title: t("Source"), skeleton: "badge" },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Source")} />,
    enableSorting: true,
    size: 140,
    minSize: 100,
    cell: ({ row }) =>
      row.original.model_info.db_model ? (
        <StatusBadge tone="info" label={t("DB Model")} />
      ) : (
        <StatusBadge tone="neutral" label={t("Config Model")} />
      ),
  },
  {
    id: "actions",
    meta: { title: t("Actions"), className: "text-right", headerClassName: "text-right" },
    header: t("Actions"),
    enableSorting: false,
    enableHiding: false,
    enableResizing: false,
    size: 110,
    minSize: 110,
    cell: ({ row }) => (
      <ModelRowActions
        model={row.original}
        userRole={userRole}
        userID={userID}
        isViewOnly={isViewOnly}
        isPausing={pausingModelId === row.original.model_info?.id}
        onDeleteClick={onDeleteClick}
        onTogglePauseClick={onTogglePauseClick}
      />
    ),
  },
];
