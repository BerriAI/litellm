import { getSpendString } from "@/utils/dataUtils";
import type { ColumnDef } from "@tanstack/react-table";
import { Badge, Button } from "@tremor/react";
import { Tooltip } from "antd";
import React, { useState } from "react";
import { getProviderLogoAndName } from "../provider_info_helpers";
import { TableHeaderSortDropdown } from "../common_components/TableHeaderSortDropdown/TableHeaderSortDropdown";
import { TimeCell } from "./time_cell";
import { MCP_CALL_TYPES } from "./constants";
import { LlmBadge, McpBadge, SparkleIcon, WrenchIcon } from "./TypeBadges";

/** API sort field mapping for /spend/logs/ui endpoint */
export const LOGS_SORT_FIELD_MAP = {
  startTime: "startTime",
  spend: "spend",
  total_tokens: "total_tokens",
} as const;

export type LogsSortField = keyof typeof LOGS_SORT_FIELD_MAP;

export interface LogsSortProps {
  sortBy: LogsSortField;
  sortOrder: "asc" | "desc";
  onSortChange: (sortBy: LogsSortField, sortOrder: "asc" | "desc") => void;
}

// Helper to get the appropriate logo URL
const getLogoUrl = (row: LogEntry, provider: string) => {
  // Check if mcp_tool_call_metadata exists and contains mcp_server_logo_url
  if (row.metadata?.mcp_tool_call_metadata?.mcp_server_logo_url) {
    return row.metadata.mcp_tool_call_metadata.mcp_server_logo_url;
  }
  // Fall back to default provider logo
  return provider ? getProviderLogoAndName(provider).logo : "";
};

export type LogEntry = {
  request_id: string;
  api_key: string;
  team_id: string;
  model: string;
  model_id: string;
  api_base?: string;
  call_type: string;
  spend: number;
  total_tokens: number;
  prompt_tokens: number;
  completion_tokens: number;
  startTime: string;
  endTime: string;
  user?: string;
  end_user?: string;
  custom_llm_provider?: string;
  metadata?: Record<string, any>;
  cache_hit: string;
  cache_key?: string;
  request_tags?: Record<string, any>;
  requester_ip_address?: string;
  messages: string | any[] | Record<string, any>;
  response: string | any[] | Record<string, any>;
  proxy_server_request?: string | any[] | Record<string, any>;
  session_id?: string;
  status?: string;
  duration?: number;
  session_total_count?: number;
  session_total_spend?: number;
  mcp_tool_call_count?: number;
  mcp_tool_call_spend?: number;
  session_llm_count?: number;
  session_mcp_count?: number;
  onKeyHashClick?: (keyHash: string) => void;
  onSessionClick?: (sessionId: string) => void;
};

const SortableHeader = ({
  label,
  field,
  sortBy,
  sortOrder,
  onSortChange,
}: {
  label: string;
  field: LogsSortField;
  sortBy: LogsSortField;
  sortOrder: "asc" | "desc";
  onSortChange: (sortBy: LogsSortField, sortOrder: "asc" | "desc") => void;
}) => (
  <div className="flex items-center gap-1">
    <span>{label}</span>
    <TableHeaderSortDropdown
      sortState={sortBy === field ? sortOrder : false}
      onSortChange={(newState) => {
        if (newState === false) {
          onSortChange("startTime", "desc");
        } else {
          onSortChange(field, newState);
        }
      }}
    />
  </div>
);

export const createColumns = (sortProps?: LogsSortProps): ColumnDef<LogEntry>[] => [
  {
    header: sortProps
      ? () => (
          <SortableHeader
            label="Time"
            field="startTime"
            sortBy={sortProps.sortBy}
            sortOrder={sortProps.sortOrder}
            onSortChange={sortProps.onSortChange}
          />
        )
      : "Time",
    accessorKey: "startTime",
    cell: (info: any) => <TimeCell utcTime={info.getValue()} />,
  },
  {
    header: "Type",
    id: "type",
    cell: (info: any) => {
      const row = info.row.original;
      const sessionCount = row.session_total_count || 1;
      const isMcp = MCP_CALL_TYPES.includes(row.call_type);
      const sessionLlmCount = row.session_llm_count ?? (isMcp ? 0 : sessionCount);
      const sessionMcpCount = row.session_mcp_count ?? (isMcp ? sessionCount : 0);

      if (isMcp) return <McpBadge />;
      if (sessionCount <= 1) return <LlmBadge />;

      // Multi-call session — show total count, plus MCP indicator when mixed.
      const sessionTypeBadge = (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-blue-50 text-blue-700 border border-blue-200 rounded-full text-[11px] font-medium whitespace-nowrap">
          <SparkleIcon />
          <span>{sessionCount}</span>
          {sessionMcpCount > 0 && (
            <>
              <span className="text-blue-300">·</span>
              <WrenchIcon />
            </>
          )}
        </span>
      );

      return (
        <Tooltip title={`${sessionLlmCount} LLM • ${sessionMcpCount} MCP`}>
          {sessionTypeBadge}
        </Tooltip>
      );
    },
  },
  {
    header: "Status",
    accessorKey: "metadata.status",
    cell: (info: any) => {
      const status = info.getValue() || "Success";
      const isSuccess = status.toLowerCase() !== "failure";

      return (
        <span
          className={`px-2 py-1 rounded-md text-xs font-medium inline-block text-center w-16 ${
            isSuccess ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
          }`}
        >
          {isSuccess ? "Success" : "Failure"}
        </span>
      );
    },
  },
  {
    header: "Session ID",
    accessorKey: "session_id",
    cell: (info: any) => {
      const value = String(info.getValue() || "");
      const onSessionClick = info.row.original.onSessionClick;
      return (
        <Tooltip title={String(info.getValue() || "")}>
          <Button
            size="xs"
            variant="light"
            className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal text-xs max-w-[15ch] truncate block"
            onClick={() => onSessionClick?.(value)}
          >
            {String(info.getValue() || "")}
          </Button>
        </Tooltip>
      );
    },
  },

  {
    header: "Request ID",
    accessorKey: "request_id",
    cell: (info: any) => (
      <Tooltip title={String(info.getValue() || "")}>
        <span className="font-mono text-xs max-w-[15ch] truncate block">{String(info.getValue() || "")}</span>
      </Tooltip>
    ),
  },
  {
    header: sortProps
      ? () => (
          <SortableHeader
            label="Cost"
            field="spend"
            sortBy={sortProps.sortBy}
            sortOrder={sortProps.sortOrder}
            onSortChange={sortProps.onSortChange}
          />
        )
      : "Cost",
    accessorKey: "spend",
    cell: (info: any) => {
      const row = info.row.original;
      const mcpCount = row.mcp_tool_call_count || 0;
      const mcpSpend = row.mcp_tool_call_spend || 0;

      return (
        <div className="flex flex-col">
          <Tooltip title={`$${String(info.getValue() || 0)}`}>
            <span>{getSpendString(info.getValue() || 0)}</span>
          </Tooltip>
          {mcpCount > 0 && mcpSpend > 0 && (
            <span className="text-[10px] text-amber-600">
              incl. {getSpendString(mcpSpend)} from {mcpCount} MCP
            </span>
          )}
        </div>
      );
    },
  },
  {
    header: "Duration (s)",
    accessorKey: "duration",
    cell: (info: any) => (
      <Tooltip title={String(info.getValue() || "-")}>
        <span className="max-w-[15ch] truncate block">{String(info.getValue() || "-")}</span>
      </Tooltip>
    ),
  },
  {
    header: "Team Name",
    accessorKey: "metadata.user_api_key_team_alias",
    cell: (info: any) => (
      <Tooltip title={String(info.getValue() || "-")}>
        <span className="max-w-[15ch] truncate block">{String(info.getValue() || "-")}</span>
      </Tooltip>
    ),
  },
  {
    header: "Key Hash",
    accessorKey: "metadata.user_api_key",
    cell: (info: any) => {
      const value = String(info.getValue() || "-");
      const onKeyHashClick = info.row.original.onKeyHashClick;

      return (
        <Tooltip title={value}>
          <span
            className="font-mono max-w-[15ch] truncate block cursor-pointer hover:text-blue-600"
            onClick={() => onKeyHashClick?.(value)}
          >
            {value}
          </span>
        </Tooltip>
      );
    },
  },
  {
    header: "Key Name",
    accessorKey: "metadata.user_api_key_alias",
    cell: (info: any) => (
      <Tooltip title={String(info.getValue() || "-")}>
        <span className="max-w-[15ch] truncate block">{String(info.getValue() || "-")}</span>
      </Tooltip>
    ),
  },
  {
    header: "Model",
    accessorKey: "model",
    cell: (info: any) => {
      const row = info.row.original;
      const provider = row.custom_llm_provider;
      const modelName = String(info.getValue() || "");
      return (
        <div className="flex items-center space-x-2">
          {provider && (
            <img
              src={getLogoUrl(row, provider)}
              alt=""
              className="w-4 h-4"
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = "none";
              }}
            />
          )}
          <Tooltip title={modelName}>
            <span className="max-w-[15ch] truncate block">{modelName}</span>
          </Tooltip>
        </div>
      );
    },
  },
  {
    header: sortProps
      ? () => (
          <SortableHeader
            label="Tokens"
            field="total_tokens"
            sortBy={sortProps.sortBy}
            sortOrder={sortProps.sortOrder}
            onSortChange={sortProps.onSortChange}
          />
        )
      : "Tokens",
    accessorKey: "total_tokens",
    cell: (info: any) => {
      const row = info.row.original;
      return (
        <span className="text-sm">
          {String(row.total_tokens || "0")}
          <span className="text-gray-400 text-xs ml-1">
            ({String(row.prompt_tokens || "0")}+{String(row.completion_tokens || "0")})
          </span>
        </span>
      );
    },
  },
  {
    header: "Internal User",
    accessorKey: "user",
    cell: (info: any) => (
      <Tooltip title={String(info.getValue() || "-")}>
        <span className="max-w-[15ch] truncate block">{String(info.getValue() || "-")}</span>
      </Tooltip>
    ),
  },
  {
    header: "End User",
    accessorKey: "end_user",
    cell: (info: any) => (
      <Tooltip title={String(info.getValue() || "-")}>
        <span className="max-w-[15ch] truncate block">{String(info.getValue() || "-")}</span>
      </Tooltip>
    ),
  },

  {
    header: "Tags",
    accessorKey: "request_tags",
    cell: (info: any) => {
      const tags = info.getValue();
      if (!tags || Object.keys(tags).length === 0) return "-";

      const tagEntries = Object.entries(tags);
      const firstTag = tagEntries[0];
      const remainingTags = tagEntries.slice(1);

      return (
        <div className="flex flex-wrap gap-1">
          <Tooltip
            title={
              <div className="flex flex-col gap-1">
                {tagEntries.map(([key, value]) => (
                  <span key={key}>
                    {key}: {String(value)}
                  </span>
                ))}
              </div>
            }
          >
            <span className="px-2 py-1 bg-gray-100 rounded-full text-xs">
              {firstTag[0]}: {String(firstTag[1])}
              {remainingTags.length > 0 && ` +${remainingTags.length}`}
            </span>
          </Tooltip>
        </div>
      );
    },
  },
];

/** Default columns without sort (for backward compatibility) */
export const columns = createColumns();

const formatMessage = (message: any): string => {
  if (!message) return "N/A";
  if (typeof message === "string") return message;
  if (typeof message === "object") {
    // Handle the {text, type} object specifically
    if (message.text) return message.text;
    if (message.content) return message.content;
    return JSON.stringify(message);
  }
  return String(message);
};

// Add this new component for displaying request/response with copy buttons
export const RequestResponsePanel = ({ request, response }: { request: any; response: any }) => {
  const requestStr = typeof request === "object" ? JSON.stringify(request, null, 2) : String(request || "{}");
  const responseStr = typeof response === "object" ? JSON.stringify(response, null, 2) : String(response || "{}");

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      console.error("Failed to copy text: ", err);
    }
  };

  return (
    <div className="grid grid-cols-2 gap-4 mt-4">
      <div className="rounded-lg border border-gray-200 bg-gray-50">
        <div className="flex justify-between items-center p-3 border-b border-gray-200">
          <h3 className="text-sm font-medium">Request</h3>
          <button
            onClick={() => copyToClipboard(requestStr)}
            className="p-1 hover:bg-gray-200 rounded"
            title="Copy request"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
          </button>
        </div>
        <pre className="p-4 overflow-auto text-xs font-mono h-64 whitespace-pre-wrap break-words">{requestStr}</pre>
      </div>

      <div className="rounded-lg border border-gray-200 bg-gray-50">
        <div className="flex justify-between items-center p-3 border-b border-gray-200">
          <h3 className="text-sm font-medium">Response</h3>
          <button
            onClick={() => copyToClipboard(responseStr)}
            className="p-1 hover:bg-gray-200 rounded"
            title="Copy response"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
          </button>
        </div>
        <pre className="p-4 overflow-auto text-xs font-mono h-64 whitespace-pre-wrap break-words">{responseStr}</pre>
      </div>
    </div>
  );
};

// New component for collapsible JSON display
const CollapsibleJsonCell = ({ jsonData }: { jsonData: any }) => {
  const [isExpanded, setIsExpanded] = React.useState(false);
  const jsonString = JSON.stringify(jsonData, null, 2);

  if (!jsonData || Object.keys(jsonData).length === 0) {
    return <span>-</span>;
  }

  return (
    <div>
      <button onClick={() => setIsExpanded(!isExpanded)} className="text-blue-500 hover:text-blue-700 text-xs">
        {isExpanded ? "Hide JSON" : "Show JSON"} ({Object.keys(jsonData).length} fields)
      </button>
      {isExpanded && (
        <pre className="mt-2 p-2 bg-gray-50 border rounded text-xs overflow-auto max-h-60">{jsonString}</pre>
      )}
    </div>
  );
};

export type AuditLogEntry = {
  id: string;
  updated_at: string;
  changed_by: string;
  changed_by_api_key: string;
  action: string;
  table_name: string;
  object_id: string;
  before_value: Record<string, any>;
  updated_values: Record<string, any>;
};

const getActionBadge = (action: string) => {
  return (
    <Badge color="gray" className="flex items-center gap-1">
      <span className="whitespace-nowrap text-xs">{action}</span>
    </Badge>
  );
};

export const auditLogColumns: ColumnDef<AuditLogEntry>[] = [
  {
    id: "expander",
    header: () => null,
    cell: ({ row }) => {
      const ExpanderCell = () => {
        const [localExpanded, setLocalExpanded] = React.useState(row.getIsExpanded());

        const toggleHandler = React.useCallback(() => {
          setLocalExpanded((prev) => !prev);
          row.getToggleExpandedHandler()();
        }, [row]);

        return row.getCanExpand() ? (
          <button
            onClick={toggleHandler}
            style={{ cursor: "pointer" }}
            aria-label={localExpanded ? "Collapse row" : "Expand row"}
            className="w-6 h-6 flex items-center justify-center focus:outline-none"
          >
            <svg
              className={`w-4 h-4 transform transition-transform ${localExpanded ? "rotate-90" : ""}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        ) : (
          <span className="w-6 h-6 flex items-center justify-center">●</span>
        );
      };
      return <ExpanderCell />;
    },
  },
  {
    header: "Timestamp",
    accessorKey: "updated_at",
    cell: (info: any) => <TimeCell utcTime={info.getValue()} />,
  },
  {
    header: "Table Name",
    accessorKey: "table_name",
    cell: (info: any) => {
      const tableName = info.getValue();
      let displayValue = tableName;
      switch (tableName) {
        case "LiteLLM_VerificationToken":
          displayValue = "Keys";
          break;
        case "LiteLLM_TeamTable":
          displayValue = "Teams";
          break;
        case "LiteLLM_OrganizationTable":
          displayValue = "Organizations";
          break;
        case "LiteLLM_UserTable":
          displayValue = "Users";
          break;
        case "LiteLLM_ProxyModelTable":
          displayValue = "Models";
          break;
        default:
          displayValue = tableName;
      }
      return <span>{displayValue}</span>;
    },
  },
  {
    header: "Action",
    accessorKey: "action",
    cell: (info: any) => <span>{getActionBadge(info.getValue())}</span>,
  },
  {
    header: "Changed By",
    accessorKey: "changed_by",
    cell: (info: any) => {
      const changedBy = info.row.original.changed_by;
      const apiKey = info.row.original.changed_by_api_key;
      return (
        <div className="space-y-1">
          <div className="font-medium">{changedBy}</div>
          {apiKey && ( // Only show API key if it exists
            <Tooltip title={apiKey}>
              <div className="text-xs text-muted-foreground max-w-[15ch] truncate">
                {" "}
                {/* Apply max-width and truncate */}
                {apiKey}
              </div>
            </Tooltip>
          )}
        </div>
      );
    },
  },
  {
    header: "Affected Item ID",
    accessorKey: "object_id",
    cell: (props) => {
      const ObjectIdDisplay = () => {
        const objectId = props.getValue();
        const [copied, setCopied] = useState(false);

        if (!objectId) return <>-</>;

        const handleCopy = async () => {
          try {
            await navigator.clipboard.writeText(String(objectId));
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          } catch (err) {
            console.error("Failed to copy object ID: ", err);
          }
        };

        return (
          <Tooltip title={copied ? "Copied!" : String(objectId)}>
            <span className="max-w-[20ch] truncate block cursor-pointer hover:text-blue-600" onClick={handleCopy}>
              {String(objectId)}
            </span>
          </Tooltip>
        );
      };
      return <ObjectIdDisplay />;
    },
  },
];
