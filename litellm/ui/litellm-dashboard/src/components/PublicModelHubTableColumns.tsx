"use client";

import { ColumnDef } from "@tanstack/react-table";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { CellTooltip, IdentityCell, StatusBadge, type StatusTone } from "@/components/shared/table_cells";
import { Badge } from "@/components/ui/badge";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { PUBLIC_MODEL_HUB_SORTABLE_FIELDS } from "@/components/publicModelHub/publicModelHubFilters";
import i18n from "@/i18n/i18n";
import { displayMode } from "@/lib/decisionModels";
import type { ModelCustomMetadataValue } from "@/components/AIHub/ModelCustomMetadata";
import { formatModelModeLabel, ProviderLogo } from "@/components/AIHub/ModelHubProviders";
import type { DeploymentPricing } from "@/components/molecules/models/deploymentPricing";

export interface ModelGroupInfo {
  model_group: string;
  providers: string[];
  max_input_tokens?: number;
  max_output_tokens?: number;
  input_cost_per_token?: number;
  output_cost_per_token?: number;
  mode?: string;
  tpm?: number;
  rpm?: number;
  supports_parallel_function_calling: boolean;
  supports_vision: boolean;
  supports_function_calling: boolean;
  supported_openai_params?: string[];
  health_status?: string;
  health_response_time?: number;
  health_checked_at?: string;
  pass_through_path?: string | null;
  pass_through_methods?: string[] | null;
  description?: string;
  custom_tags?: string[] | null;
  custom_metadata?: Record<string, ModelCustomMetadataValue> | null;
  deployment_pricing?: DeploymentPricing[] | null;
  [key: string]: any;
}

export interface AgentCard {
  protocolVersion: string;
  name: string;
  description: string;
  url: string;
  version: string;
  capabilities?: {
    streaming?: boolean;
    pushNotifications?: boolean;
    stateTransitionHistory?: boolean;
  };
  defaultInputModes: string[];
  defaultOutputModes: string[];
  skills: Array<{
    id: string;
    name: string;
    description: string;
    tags: string[];
  }>;
  iconUrl?: string;
  provider?: {
    organization: string;
    url: string;
  };
  documentationUrl?: string;
  [key: string]: any;
}

export interface MCPServerData {
  server_id: string;
  name: string;
  alias?: string | null;
  server_name: string;
  transport: string;
  spec_path?: string | null;
  auth_type: string;
  mcp_info: {
    server_name: string;
    description?: string;
    mcp_server_cost_info?: any;
  };
  [key: string]: any;
}

const formatCapabilityName = (key: string) =>
  key
    .replace(/^supports_/, "")
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");

const formatCost = (cost: number) => `$${(cost * 1_000_000).toFixed(4)}`;

const tokenCostLabel = (model: ModelGroupInfo, cost: number | null | undefined): string => {
  if (model.pass_through_path) return "n/a";
  return cost ? formatCost(cost) : i18n.t("aiHub.publicModelHubColumns.free");
};

const formatTokens = (tokens: number | undefined) => {
  if (!tokens) return i18n.t("aiHub.publicModelHubColumns.notAvailable");
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(0)}K`;
  return tokens.toString();
};

const formatLimits = (rpm?: number, tpm?: number) => {
  const limits = [
    ...(rpm ? [i18n.t("aiHub.publicModelHubColumns.rpmLimit", { value: rpm.toLocaleString() })] : []),
    ...(tpm ? [i18n.t("aiHub.publicModelHubColumns.tpmLimit", { value: tpm.toLocaleString() })] : []),
  ];
  return limits.length > 0 ? limits.join(", ") : i18n.t("aiHub.publicModelHubColumns.notAvailable");
};

const getModeIcon = (mode: string) => {
  switch (mode?.toLowerCase()) {
    case "chat":
      return "💬";
    case "rerank":
      return "🔄";
    case "embedding":
      return "📄";
    case "passthrough":
      return "🔀";
    default:
      return "🤖";
  }
};

const HEALTH_TONES: Record<string, StatusTone> = {
  healthy: "success",
  unhealthy: "error",
};

function ProviderChips({ providers }: { providers: string[] }) {
  return (
    <div className="flex flex-wrap gap-1">
      {providers.map((provider) => {
        const { logo } = getProviderLogoAndName(provider);
        return (
          <span key={provider} className="flex items-center gap-1 rounded-md bg-muted px-2 py-1 text-xs">
            {logo && (
              <img
                src={logo}
                alt={provider}
                className="size-3 shrink-0 object-contain"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = "none";
                }}
              />
            )}
            <span className="capitalize">{provider}</span>
          </span>
        );
      })}
    </div>
  );
}

function OverflowChips({ items }: { items: string[] }) {
  if (items.length === 0) {
    return <span className="text-xs text-muted-foreground">-</span>;
  }
  return (
    <div className="flex items-center gap-1">
      <Badge variant="secondary">{items[0]}</Badge>
      {items.length > 1 && (
        <CellTooltip
          content={
            <div className="space-y-1">
              {items.map((item) => (
                <div key={item} className="text-xs">
                  • {item}
                </div>
              ))}
            </div>
          }
          trigger={<span className="cursor-default text-xs text-muted-foreground">+{items.length - 1}</span>}
        />
      )}
    </div>
  );
}

interface PublicModelHubColumnsDeps {
  onModelClick: (model: ModelGroupInfo) => void;
  showProviderLogos?: boolean;
}

export const getPublicModelHubColumns = ({
  onModelClick,
  showProviderLogos = false,
}: PublicModelHubColumnsDeps): ColumnDef<ModelGroupInfo>[] => {
  const columns: ColumnDef<ModelGroupInfo>[] = [
    {
      id: "model_group",
      accessorKey: "model_group",
      meta: {
        title: showProviderLogos
          ? i18n.t("aiHub.publicModelHubColumns.models.modelChat")
          : i18n.t("aiHub.publicModelHubColumns.models.modelName"),
      },
      header: ({ column }) => (
        <DataTableSortHeader
          column={column}
          title={
            showProviderLogos
              ? i18n.t("aiHub.publicModelHubColumns.models.modelChat")
              : i18n.t("aiHub.publicModelHubColumns.models.modelName")
          }
        />
      ),
      size: 200,
      sortingFn: "alphanumeric",
      cell: ({ row }) => (
        <IdentityCell
          title={
            showProviderLogos && row.original.providers?.[0] ? (
              <span className="flex min-w-0 items-center gap-2">
                <ProviderLogo provider={row.original.providers[0]} className="chat-model-provider-logo-small" />
                <span className="flex min-w-0 flex-col gap-0.5">
                  <span className="truncate">{row.original.model_group}</span>{" "}
                  <span className="truncate font-sans text-[11px] font-normal text-muted-foreground">
                    {(row.original.providers ?? [])
                      .map((provider) => getProviderLogoAndName(provider).displayName)
                      .join(", ")}
                  </span>
                </span>
              </span>
            ) : (
              row.original.model_group
            )
          }
          titleClassName="font-mono text-xs font-normal"
          className="max-w-72"
          onClick={() => onModelClick(row.original)}
        />
      ),
    },
    {
      id: "providers",
      accessorKey: "providers",
      meta: { title: i18n.t("aiHub.publicModelHubColumns.models.providers"), skeleton: "chips" },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={i18n.t("aiHub.publicModelHubColumns.models.providers")} />
      ),
      size: 150,
      sortingFn: (rowA, rowB) =>
        (rowA.original.providers ?? []).join(", ").localeCompare((rowB.original.providers ?? []).join(", ")),
      cell: ({ row }) => <ProviderChips providers={row.original.providers ?? []} />,
    },
    {
      id: "mode",
      accessorKey: "mode",
      meta: { title: i18n.t("aiHub.publicModelHubColumns.models.mode") },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={i18n.t("aiHub.publicModelHubColumns.models.mode")} />
      ),
      size: 110,
      sortingFn: "alphanumeric",
      cell: ({ row }) => (
        <span className="flex items-center gap-2 text-sm">
          {!showProviderLogos && <span>{getModeIcon(row.original.mode || "")}</span>}
          <span>
            {showProviderLogos
              ? formatModelModeLabel(row.original.mode)
              : displayMode(row.original.mode) ?? i18n.t("aiHub.publicModelHubColumns.chatMode")}
          </span>
        </span>
      ),
    },
    {
      id: "max_input_tokens",
      accessorKey: "max_input_tokens",
      meta: { title: i18n.t("aiHub.publicModelHubColumns.models.maxInput"), numeric: true },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={i18n.t("aiHub.publicModelHubColumns.models.maxInput")} />
      ),
      size: 100,
      cell: ({ row }) => <span className="text-sm">{formatTokens(row.original.max_input_tokens)}</span>,
    },
    {
      id: "max_output_tokens",
      accessorKey: "max_output_tokens",
      meta: { title: i18n.t("aiHub.publicModelHubColumns.models.maxOutput"), numeric: true },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={i18n.t("aiHub.publicModelHubColumns.models.maxOutput")} />
      ),
      size: 100,
      cell: ({ row }) => <span className="text-sm">{formatTokens(row.original.max_output_tokens)}</span>,
    },
    {
      id: "input_cost_per_token",
      accessorKey: "input_cost_per_token",
      meta: {
        title: showProviderLogos
          ? i18n.t("aiHub.publicModelHubColumns.models.costChat")
          : i18n.t("aiHub.publicModelHubColumns.models.inputCost"),
        numeric: true,
      },
      header: ({ column }) => (
        <DataTableSortHeader
          column={column}
          title={
            showProviderLogos
              ? i18n.t("aiHub.publicModelHubColumns.models.costChat")
              : i18n.t("aiHub.publicModelHubColumns.models.inputCost")
          }
        />
      ),
      size: 110,
      cell: ({ row }) =>
        showProviderLogos ? (
          <div className="flex flex-col gap-0.5 text-xs tabular-nums">
            <span>
              {i18n.t("aiHub.publicModelHubColumns.inputCost", {
                cost: tokenCostLabel(row.original, row.original.input_cost_per_token),
              })}
            </span>
            <span className="text-muted-foreground">
              {i18n.t("aiHub.publicModelHubColumns.outputCost", {
                cost: tokenCostLabel(row.original, row.original.output_cost_per_token),
              })}
            </span>
          </div>
        ) : (
          <span className="text-sm">{tokenCostLabel(row.original, row.original.input_cost_per_token)}</span>
        ),
    },
    {
      id: "output_cost_per_token",
      accessorKey: "output_cost_per_token",
      meta: { title: i18n.t("aiHub.publicModelHubColumns.models.outputCost"), numeric: true },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={i18n.t("aiHub.publicModelHubColumns.models.outputCost")} />
      ),
      size: 110,
      cell: ({ row }) => (
        <span className="text-sm">{tokenCostLabel(row.original, row.original.output_cost_per_token)}</span>
      ),
    },
    {
      id: "features",
      meta: { title: i18n.t("aiHub.publicModelHubColumns.models.features"), skeleton: "chips" },
      header: i18n.t("aiHub.publicModelHubColumns.models.features"),
      size: 140,
      cell: ({ row }) => {
        const features = Object.entries(row.original)
          .filter(([key, value]) => key.startsWith("supports_") && value === true)
          .map(([key]) => formatCapabilityName(key));
        return <OverflowChips items={features} />;
      },
    },
    {
      id: "custom_tags",
      meta: { title: i18n.t("aiHub.publicModelHubColumns.models.tags"), skeleton: "chips" },
      header: i18n.t("aiHub.publicModelHubColumns.models.tags"),
      size: 140,
      cell: ({ row }) => <OverflowChips items={row.original.custom_tags ?? []} />,
    },
    {
      id: "health_status",
      accessorKey: "health_status",
      meta: { title: i18n.t("aiHub.publicModelHubColumns.models.healthStatus"), skeleton: "badge" },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={i18n.t("aiHub.publicModelHubColumns.models.healthStatus")} />
      ),
      size: 130,
      cell: ({ row }) => {
        const model = row.original;
        const responseTimeLabel = model.health_response_time
          ? i18n.t("aiHub.publicModelHubColumns.responseTime", { ms: Number(model.health_response_time).toFixed(2) })
          : i18n.t("aiHub.publicModelHubColumns.notAvailable");
        const lastCheckedLabel = model.health_checked_at
          ? i18n.t("aiHub.publicModelHubColumns.lastChecked", {
              date: new Date(model.health_checked_at).toLocaleString(),
            })
          : i18n.t("aiHub.publicModelHubColumns.notAvailable");
        return (
          <CellTooltip
            content={
              <>
                <div>{responseTimeLabel}</div>
                <div>{lastCheckedLabel}</div>
              </>
            }
            trigger={
              <span className="capitalize">
                <StatusBadge
                  tone={HEALTH_TONES[model.health_status ?? ""] || "neutral"}
                  label={model.health_status ?? i18n.t("aiHub.publicModelHubColumns.healthUnknown")}
                />
              </span>
            }
          />
        );
      },
    },
    {
      id: "rpm",
      accessorKey: "rpm",
      meta: { title: i18n.t("aiHub.publicModelHubColumns.models.limits") },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={i18n.t("aiHub.publicModelHubColumns.models.limits")} />
      ),
      size: 150,
      cell: ({ row }) => (
        <span className="text-xs text-muted-foreground">{formatLimits(row.original.rpm, row.original.tpm)}</span>
      ),
    },
  ];
  return columns
    .filter(
      (column) =>
        !showProviderLogos || !["providers", "custom_tags", "output_cost_per_token"].includes(String(column.id)),
    )
    .map((column) => ({
      ...column,
      meta: {
        ...column.meta,
        className:
          showProviderLogos && !["model_group", "input_cost_per_token"].includes(String(column.id))
            ? "hidden md:table-cell"
            : column.meta?.className,
      },
      enableSorting: PUBLIC_MODEL_HUB_SORTABLE_FIELDS.includes(String(column.id)),
    }));
};

interface PublicAgentHubColumnsDeps {
  onAgentClick: (agent: AgentCard) => void;
}

export const getPublicAgentHubColumns = ({ onAgentClick }: PublicAgentHubColumnsDeps): ColumnDef<AgentCard>[] => [
  {
    id: "name",
    accessorKey: "name",
    meta: { title: i18n.t("aiHub.publicModelHubColumns.agents.name") },
    header: ({ column }) => (
      <DataTableSortHeader column={column} title={i18n.t("aiHub.publicModelHubColumns.agents.name")} />
    ),
    size: 200,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => (
      <IdentityCell
        title={row.original.name}
        titleClassName="font-mono text-xs font-normal"
        className="max-w-72"
        onClick={() => onAgentClick(row.original)}
      />
    ),
  },
  {
    id: "description",
    accessorKey: "description",
    meta: { title: i18n.t("aiHub.publicModelHubColumns.agents.description") },
    header: i18n.t("aiHub.publicModelHubColumns.agents.description"),
    size: 260,
    enableSorting: false,
    cell: ({ row }) => (
      <span className="block max-w-72 truncate text-sm" title={row.original.description || undefined}>
        {row.original.description || "-"}
      </span>
    ),
  },
  {
    id: "version",
    accessorKey: "version",
    meta: { title: i18n.t("aiHub.publicModelHubColumns.agents.version") },
    header: ({ column }) => (
      <DataTableSortHeader column={column} title={i18n.t("aiHub.publicModelHubColumns.agents.version")} />
    ),
    size: 90,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => <span className="text-sm">{row.original.version}</span>,
  },
  {
    id: "provider",
    meta: { title: i18n.t("aiHub.publicModelHubColumns.agents.provider") },
    header: i18n.t("aiHub.publicModelHubColumns.agents.provider"),
    size: 130,
    enableSorting: false,
    cell: ({ row }) =>
      row.original.provider ? (
        <span className="text-sm font-medium">{row.original.provider.organization}</span>
      ) : (
        <span className="text-xs text-muted-foreground">-</span>
      ),
  },
  {
    id: "skills",
    meta: { title: i18n.t("aiHub.publicModelHubColumns.agents.skills"), skeleton: "chips" },
    header: i18n.t("aiHub.publicModelHubColumns.agents.skills"),
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <OverflowChips items={(row.original.skills || []).map((skill) => skill.name)} />,
  },
  {
    id: "capabilities",
    meta: { title: i18n.t("aiHub.publicModelHubColumns.agents.capabilities"), skeleton: "chips" },
    header: i18n.t("aiHub.publicModelHubColumns.agents.capabilities"),
    size: 160,
    enableSorting: false,
    cell: ({ row }) => {
      const capabilityList = Object.entries(row.original.capabilities || {})
        .filter(([, value]) => value === true)
        .map(([key]) => key);
      if (capabilityList.length === 0) {
        return <span className="text-xs text-muted-foreground">-</span>;
      }
      return (
        <div className="flex flex-wrap gap-1">
          {capabilityList.map((capability) => (
            <Badge key={capability} variant="outline" className="capitalize">
              {capability}
            </Badge>
          ))}
        </div>
      );
    },
  },
];

interface PublicMCPHubColumnsDeps {
  onServerClick: (server: MCPServerData) => void;
}

export const getPublicMCPHubColumns = ({ onServerClick }: PublicMCPHubColumnsDeps): ColumnDef<MCPServerData>[] => [
  {
    id: "server_name",
    accessorKey: "server_name",
    meta: { title: i18n.t("aiHub.publicModelHubColumns.mcp.serverName") },
    header: ({ column }) => (
      <DataTableSortHeader column={column} title={i18n.t("aiHub.publicModelHubColumns.mcp.serverName")} />
    ),
    size: 180,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => (
      <IdentityCell
        title={row.original.server_name}
        titleClassName="font-mono text-xs font-normal"
        className="max-w-72"
        onClick={() => onServerClick(row.original)}
      />
    ),
  },
  {
    id: "description",
    meta: { title: i18n.t("aiHub.publicModelHubColumns.mcp.description") },
    header: i18n.t("aiHub.publicModelHubColumns.mcp.description"),
    size: 260,
    enableSorting: false,
    cell: ({ row }) => {
      const description = String(row.original.mcp_info?.description ?? "-");
      return (
        <span className="block max-w-72 truncate text-sm" title={description}>
          {description}
        </span>
      );
    },
  },
  {
    id: "transport",
    accessorKey: "transport",
    meta: { title: i18n.t("aiHub.publicModelHubColumns.mcp.transport"), skeleton: "badge" },
    header: ({ column }) => (
      <DataTableSortHeader column={column} title={i18n.t("aiHub.publicModelHubColumns.mcp.transport")} />
    ),
    size: 110,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => (
      <Badge variant="secondary" className="font-mono font-normal uppercase">
        {row.original.transport}
      </Badge>
    ),
  },
  {
    id: "auth_type",
    accessorKey: "auth_type",
    meta: { title: i18n.t("aiHub.publicModelHubColumns.mcp.authType"), skeleton: "badge" },
    header: ({ column }) => (
      <DataTableSortHeader column={column} title={i18n.t("aiHub.publicModelHubColumns.mcp.authType")} />
    ),
    size: 110,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => (
      <StatusBadge tone={row.original.auth_type === "none" ? "neutral" : "success"} label={row.original.auth_type} />
    ),
  },
];
