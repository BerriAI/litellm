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
      return row.getCanExpand() ? (
        <button
          {...{
            onClick: row.getToggleExpandedHandler(),
            style: { cursor: "pointer" },
          }}
        >
          {row.getIsExpanded() ? "▼" : "▶"}
        </button>
      ) : (
        "●"
      );
    },
  },
  {
    header: "Time",
    accessorKey: "startTime",
    cell: (info: any) => <TimeCell utcTime={info.getValue()} />,
  },
  {
    header: "Request ID",
    accessorKey: "request_id",
    cell: (info: any) => (
      <Tooltip title={String(info.getValue() || "")}>
        <span className="font-mono text-xs max-w-[100px] truncate block">
          {String(info.getValue() || "")}
        </span>
      </Tooltip>
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
    cell: (info: any) => <span>{String(info.getValue() || "-")}</span>,
  },
  {
    header: "Key Name",
    accessorKey: "metadata.user_api_key_alias",
    cell: (info: any) => <span>{String(info.getValue() || "-")}</span>,
  },
  {
    header: "Request",
    accessorKey: "messages",
    cell: (info: any) => {
      const messages = info.getValue();
      try {
        const content =
          typeof messages === "string" ? JSON.parse(messages) : messages;
        let displayText = "";

        if (Array.isArray(content)) {
          displayText = formatMessage(content[0]?.content);
        } else {
          displayText = formatMessage(content);
        }

        return <span className="truncate max-w-md text-sm">{displayText}</span>;
      } catch (e) {
        return (
          <span className="truncate max-w-md text-sm">
            {formatMessage(messages)}
          </span>
        );
      }
    },
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
            <span className="max-w-[100px] truncate">
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
    cell: (info: any) => <span>{String(info.getValue() || "-")}</span>,
  },
  {
    header: "End User",
    accessorKey: "end_user",
    cell: (info: any) => <span>{String(info.getValue() || "-")}</span>,
  },
  {
    header: "Cost",
    accessorKey: "spend",
    cell: (info: any) => (
      <span>${Number(info.getValue() || 0).toFixed(6)}</span>
    ),
  },
  {
    header: "Tags",
    accessorKey: "request_tags",
    cell: (info: any) => {
      const tags = info.getValue();
      if (!tags || Object.keys(tags).length === 0) return "-";
      return (
        <div className="flex flex-wrap gap-1">
          {Object.entries(tags).map(([key, value]) => (
            <span
              key={key}
              className="px-2 py-1 bg-gray-100 rounded-full text-xs"
            >
              {key}: {String(value)}
            </span>
          ))}
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
