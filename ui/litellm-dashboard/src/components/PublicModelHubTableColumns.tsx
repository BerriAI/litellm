"use client";

import { ColumnDef } from "@tanstack/react-table";
import { englishTranslate, type Translate } from "@/i18n/language";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { CellTooltip, IdentityCell, StatusBadge, type StatusTone } from "@/components/shared/table_cells";
import { Badge } from "@/components/ui/badge";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { PUBLIC_MODEL_HUB_SORTABLE_FIELDS } from "@/components/publicModelHub/publicModelHubFilters";
import { displayMode } from "@/lib/decisionModels";
import type { ModelCustomMetadataValue } from "@/components/AIHub/ModelCustomMetadata";
import { formatModelModeLabel, ProviderLogo } from "@/components/AIHub/ModelHubProviders";
import { translateModelLabel } from "@/lib/modelLabels";

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
  description?: string;
  custom_tags?: string[] | null;
  custom_metadata?: Record<string, ModelCustomMetadataValue> | null;
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

const formatTokens = (tokens: number | undefined, t: Translate) => {
  if (!tokens) return t("N/A");
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(0)}K`;
  return tokens.toString();
};

const formatLimits = (rpm: number | undefined, tpm: number | undefined, t: Translate) => {
  const limits = [...(rpm ? [`RPM: ${rpm.toLocaleString()}`] : []), ...(tpm ? [`TPM: ${tpm.toLocaleString()}`] : [])];
  return limits.length > 0 ? limits.join(", ") : t("N/A");
};

const getModeIcon = (mode: string) => {
  switch (mode?.toLowerCase()) {
    case "chat":
      return "💬";
    case "rerank":
      return "🔄";
    case "embedding":
      return "📄";
    default:
      return "🤖";
  }
};

const HEALTH_TONES: Record<string, StatusTone> = {
  healthy: "success",
  unhealthy: "error",
};

// 只翻译约定的健康状态，保留外部扩展状态的原始值。
const healthStatusLabel = (status: string | undefined, t: Translate): string => {
  if (status === "healthy") return t("healthy");
  if (status === "unhealthy") return t("unhealthy");
  return status ?? t("Unknown");
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
  t?: Translate;
  onModelClick: (model: ModelGroupInfo) => void;
  showProviderLogos?: boolean;
}

export const getPublicModelHubColumns = ({
  onModelClick,
  showProviderLogos = false,
  t = englishTranslate,
}: PublicModelHubColumnsDeps): ColumnDef<ModelGroupInfo>[] => {
  const columns: ColumnDef<ModelGroupInfo>[] = [
    {
      id: "model_group",
      accessorKey: "model_group",
      meta: { title: showProviderLogos ? t("Model") : t("Model Name") },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={showProviderLogos ? t("Model") : t("Model Name")} />
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
      meta: { title: t("Providers"), skeleton: "chips" },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("Providers")} />,
      size: 150,
      sortingFn: (rowA, rowB) =>
        (rowA.original.providers ?? []).join(", ").localeCompare((rowB.original.providers ?? []).join(", ")),
      cell: ({ row }) => <ProviderChips providers={row.original.providers ?? []} />,
    },
    {
      id: "mode",
      accessorKey: "mode",
      meta: { title: t("Mode") },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("Mode")} />,
      size: 110,
      sortingFn: "alphanumeric",
      cell: ({ row }) => (
        <span className="flex items-center gap-2 text-sm">
          {!showProviderLogos && <span>{getModeIcon(row.original.mode || "")}</span>}
          <span>
            {row.original.mode == null
              ? t("Chat")
              : translateModelLabel(
                  displayMode(row.original.mode) ?? "",
                  "mode",
                  t,
                  showProviderLogos ? formatModelModeLabel(row.original.mode) : displayMode(row.original.mode),
                )}
          </span>
        </span>
      ),
    },
    {
      id: "max_input_tokens",
      accessorKey: "max_input_tokens",
      meta: { title: t("Max Input"), numeric: true },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("Max Input")} />,
      size: 100,
      cell: ({ row }) => <span className="text-sm">{formatTokens(row.original.max_input_tokens, t)}</span>,
    },
    {
      id: "max_output_tokens",
      accessorKey: "max_output_tokens",
      meta: { title: t("Max Output"), numeric: true },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("Max Output")} />,
      size: 100,
      cell: ({ row }) => <span className="text-sm">{formatTokens(row.original.max_output_tokens, t)}</span>,
    },
    {
      id: "input_cost_per_token",
      accessorKey: "input_cost_per_token",
      meta: { title: showProviderLogos ? "Cost / 1M tokens" : "Input $/1M", numeric: true },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={showProviderLogos ? t("Cost / 1M tokens") : t("Input $/1M")} />
      ),
      size: 110,
      cell: ({ row }) =>
        showProviderLogos ? (
          <div className="flex flex-col gap-0.5 text-xs tabular-nums">
            <span>
              {t("Input")}{" "}
              {row.original.input_cost_per_token ? formatCost(row.original.input_cost_per_token) : t("Free")}
            </span>
            <span className="text-muted-foreground">
              {t("Output")}{" "}
              {row.original.output_cost_per_token ? formatCost(row.original.output_cost_per_token) : t("Free")}
            </span>
          </div>
        ) : (
          <span className="text-sm">
            {row.original.input_cost_per_token ? formatCost(row.original.input_cost_per_token) : t("Free")}
          </span>
        ),
    },
    {
      id: "output_cost_per_token",
      accessorKey: "output_cost_per_token",
      meta: { title: t("Output $/1M"), numeric: true },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("Output $/1M")} />,
      size: 110,
      cell: ({ row }) => (
        <span className="text-sm">
          {row.original.output_cost_per_token ? formatCost(row.original.output_cost_per_token) : t("Free")}
        </span>
      ),
    },
    {
      id: "features",
      meta: { title: t("Features"), skeleton: "chips" },
      header: t("Features"),
      size: 140,
      cell: ({ row }) => {
        const features = Object.entries(row.original)
          .filter(([key, value]) => key.startsWith("supports_") && value === true)
          .map(([key]) => translateModelLabel(formatCapabilityName(key), "feature", t));
        return <OverflowChips items={features} />;
      },
    },
    {
      id: "custom_tags",
      meta: { title: t("Tags"), skeleton: "chips" },
      header: t("Tags"),
      size: 140,
      cell: ({ row }) => <OverflowChips items={row.original.custom_tags ?? []} />,
    },
    {
      id: "health_status",
      accessorKey: "health_status",
      meta: { title: t("Health Status"), skeleton: "badge" },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("Health Status")} />,
      size: 130,
      cell: ({ row }) => {
        const model = row.original;
        const responseTimeLabel = model.health_response_time
          ? t("Response Time: {time}ms", { time: Number(model.health_response_time).toFixed(2) })
          : t("N/A");
        const lastCheckedLabel = model.health_checked_at
          ? t("Last Checked: {time}", { time: new Date(model.health_checked_at).toLocaleString() })
          : t("N/A");
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
                  label={healthStatusLabel(model.health_status, t)}
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
      meta: { title: t("Limits") },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("Limits")} />,
      size: 150,
      cell: ({ row }) => (
        <span className="text-xs text-muted-foreground">{formatLimits(row.original.rpm, row.original.tpm, t)}</span>
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
  t?: Translate;
  onAgentClick: (agent: AgentCard) => void;
}

export const getPublicAgentHubColumns = ({
  onAgentClick,
  t = englishTranslate,
}: PublicAgentHubColumnsDeps): ColumnDef<AgentCard>[] => [
  {
    id: "name",
    accessorKey: "name",
    meta: { title: t("Agent Name") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Agent Name")} />,
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
    meta: { title: t("Description") },
    header: t("Description"),
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
    meta: { title: t("Version") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Version")} />,
    size: 90,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => <span className="text-sm">{row.original.version}</span>,
  },
  {
    id: "provider",
    meta: { title: t("Provider") },
    header: t("Provider"),
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
    meta: { title: t("Skills"), skeleton: "chips" },
    header: t("Skills"),
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <OverflowChips items={(row.original.skills || []).map((skill) => skill.name)} />,
  },
  {
    id: "capabilities",
    meta: { title: t("Capabilities"), skeleton: "chips" },
    header: t("Capabilities"),
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
  t?: Translate;
  onServerClick: (server: MCPServerData) => void;
}

export const getPublicMCPHubColumns = ({
  onServerClick,
  t = englishTranslate,
}: PublicMCPHubColumnsDeps): ColumnDef<MCPServerData>[] => [
  {
    id: "server_name",
    accessorKey: "server_name",
    meta: { title: t("Server Name") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Server Name")} />,
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
    meta: { title: t("Description") },
    header: t("Description"),
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
    meta: { title: t("Transport"), skeleton: "badge" },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Transport")} />,
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
    meta: { title: t("Auth Type"), skeleton: "badge" },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("Auth Type")} />,
    size: 110,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => (
      <StatusBadge tone={row.original.auth_type === "none" ? "neutral" : "success"} label={row.original.auth_type} />
    ),
  },
];
