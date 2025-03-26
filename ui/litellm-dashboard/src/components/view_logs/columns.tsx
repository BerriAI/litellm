import type { ColumnDef } from "@tanstack/react-table";
import { getCountryFromIP } from "./ip_lookup";
import moment from "moment";
import React from "react";
import { CountryCell } from "./country_cell";
import { getProviderLogoAndName } from "../provider_info_helpers";
import { Tooltip } from "antd";
import { TimeCell } from "./time_cell";

export type LogEntry = {
  request_id: string;
  api_key: string;
  team_id: string;
  model: string;
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
};

export const columns: ColumnDef<LogEntry>[] = [
  {
    id: "expander",
    header: () => null,
    cell: ({ row }) => {
      // Convert the cell function to a React component to properly use hooks
      const ExpanderCell = () => {
        const [localExpanded, setLocalExpanded] = React.useState(row.getIsExpanded());

        // Memoize the toggle handler to prevent unnecessary re-renders
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
              className={`w-4 h-4 transform transition-transform duration-75 ${
                localExpanded ? 'rotate-90' : ''
              }`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 5l7 7-7 7"
              />
            </svg>
          </button>
        ) : (
          <span className="w-6 h-6 flex items-center justify-center">●</span>
        );
      };

      // Return the component
      return <ExpanderCell />;
    },
  },
  {
    header: "Time",
    accessorKey: "startTime",
    cell: (info: any) => <TimeCell utcTime={info.getValue()} />,
  },
  {
    header: "Status",
    accessorKey: "metadata.status",
    cell: (info: any) => {
      const status = info.getValue() || "Success";
      const isSuccess = status.toLowerCase() !== "failure";
      
      return (
        <span className={`px-2 py-1 rounded-md text-xs font-medium inline-block text-center w-16 ${
          isSuccess 
            ? 'bg-green-100 text-green-800' 
            : 'bg-red-100 text-red-800'
        }`}>
          {isSuccess ? "Success" : "Failure"}
        </span>
      );
    },
  },
  {
    header: "Request ID",
    accessorKey: "request_id",
    cell: (info: any) => (
      <Tooltip title={String(info.getValue() || "")}>
        <span className="font-mono text-xs max-w-[15ch] truncate block">
          {String(info.getValue() || "")}
        </span>
      </Tooltip>
    ),
  },
  {
    header: "Cost",
    accessorKey: "spend",
    cell: (info: any) => (
      <span>${Number(info.getValue() || 0).toFixed(6)}</span>
    ),
  },
  {
    header: "Country",
    accessorKey: "requester_ip_address",
    cell: (info: any) => <CountryCell ipAddress={info.getValue()} />,
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
      return (
        <Tooltip title={value}>
          <span className="font-mono max-w-[15ch] truncate block">{value}</span>
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
              src={getProviderLogoAndName(provider).logo}
              alt=""
              className="w-4 h-4"
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
              }}
            />
          )}
          <Tooltip title={modelName}>
            <span className="max-w-[15ch] truncate block">
              {modelName}
            </span>
          </Tooltip>
        </div>
      );
    },
  },
  {
    header: "Tokens",
    accessorKey: "total_tokens",
    cell: (info: any) => {
      const row = info.row.original;
      return (
        <span className="text-sm">
          {String(row.total_tokens || "0")}
          <span className="text-gray-400 text-xs ml-1">
            ({String(row.prompt_tokens || "0")}+
            {String(row.completion_tokens || "0")})
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
  const requestStr = typeof request === 'object' ? JSON.stringify(request, null, 2) : String(request || '{}');
  const responseStr = typeof response === 'object' ? JSON.stringify(response, null, 2) : String(response || '{}');
  
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
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
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
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
