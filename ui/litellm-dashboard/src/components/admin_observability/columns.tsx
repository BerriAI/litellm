import type { ColumnsType } from "antd/es/table";
import { Tooltip } from "antd";
import { getSpendString } from "@/utils/dataUtils";
import { TimeCell } from "@/components/view_logs/time_cell";
import type { AdminObservabilityRow } from "./types";

const TruncatedCell = ({ value, maxWidth = 120 }: { value: string; maxWidth?: number }) => (
  <Tooltip title={value || "-"}>
    <span className="truncate block" style={{ maxWidth }}>
      {value || "-"}
    </span>
  </Tooltip>
);

export const createAdminObservabilityColumns = (): ColumnsType<AdminObservabilityRow> => [
  {
    title: "Time",
    dataIndex: "startTime",
    key: "startTime",
    width: 180,
    render: (value: string) => <TimeCell utcTime={value} />,
  },
  {
    title: "User",
    dataIndex: "user",
    key: "user",
    width: 150,
    render: (value: string) => <TruncatedCell value={value} />,
  },
  {
    title: "Model",
    dataIndex: "model",
    key: "model",
    width: 180,
    render: (value: string) => <TruncatedCell value={value} maxWidth={170} />,
  },
  {
    title: "Status",
    key: "status",
    width: 100,
    render: (_: unknown, row: AdminObservabilityRow) => {
      const status = row.metadata?.status || row.status || "Success";
      const isSuccess = String(status).toLowerCase() !== "failure";
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
    title: "Input Tokens",
    dataIndex: "prompt_tokens",
    key: "prompt_tokens",
    width: 110,
    align: "right",
    render: (value: number | undefined) => <span className="text-sm">{value ?? 0}</span>,
  },
  {
    title: "Output Tokens",
    dataIndex: "completion_tokens",
    key: "completion_tokens",
    width: 110,
    align: "right",
    render: (value: number | undefined) => <span className="text-sm">{value ?? 0}</span>,
  },
  {
    title: "Total Tokens",
    dataIndex: "total_tokens",
    key: "total_tokens",
    width: 120,
    align: "right",
    render: (value: number | undefined) => <span className="text-sm font-medium">{value ?? 0}</span>,
  },
  {
    title: "Tool Calls",
    key: "tool_calls",
    width: 170,
    render: (_: unknown, row: AdminObservabilityRow) => {
      const mcpName = row.metadata?.mcp_tool_call_metadata?.namespaced_tool_name;
      if (mcpName) {
        return (
          <Tooltip title={mcpName}>
            <span className="truncate block text-amber-700" style={{ maxWidth: 160 }}>
              {mcpName}
            </span>
          </Tooltip>
        );
      }

      const toolCalls = row.metadata?.mcp_tool_call_metadata?.tool_calls ?? [];
      const count = Array.isArray(toolCalls) ? toolCalls.length : 0;
      if (count === 0) return <span className="text-gray-400">-</span>;

      function toolName(tc: unknown): string {
        if (tc === null || typeof tc !== "object") return "tool";
        const record = tc as Record<string, unknown>;
        const fn = record.function;
        if (fn !== null && typeof fn === "object") {
          const fnRecord = fn as Record<string, unknown>;
          if (typeof fnRecord.name === "string" && fnRecord.name) return fnRecord.name;
        }
        if (typeof record.name === "string" && record.name) return record.name;
        return "tool";
      }

      const names = toolCalls.map(toolName).filter(Boolean).join(", ");
      const label = `${count} call${count > 1 ? "s" : ""}${names ? `: ${names}` : ""}`;
      return (
        <Tooltip title={names || label}>
          <span className="truncate block" style={{ maxWidth: 160 }}>
            {label}
          </span>
        </Tooltip>
      );
    },
  },
  {
    title: "Cost",
    dataIndex: "spend",
    key: "spend",
    width: 100,
    render: (value: number | undefined) => (
      <Tooltip title={`$${String(value ?? 0)}`}>
        <span>{getSpendString(value ?? 0)}</span>
      </Tooltip>
    ),
  },
  {
    title: "Request ID",
    dataIndex: "request_id",
    key: "request_id",
    width: 170,
    render: (value: string) => <TruncatedCell value={value} maxWidth={160} />,
  },
];

export const adminObservabilityColumns = createAdminObservabilityColumns();
