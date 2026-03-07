import React, { useState, useEffect, useMemo } from "react";
import { Card, Title, Text } from "@tremor/react";
import { Table, Tag, Tooltip, Pagination, Empty, Spin, Select } from "antd";
import moment from "moment";
import { fetchMCPServerLogs } from "../networking";

interface MCPServerLogsProps {
  serverId: string;
  accessToken: string | null;
}

interface MCPLogEntry {
  request_id: string;
  call_type: string;
  mcp_namespaced_tool_name: string | null;
  status: string | null;
  spend: number;
  total_tokens: number | null;
  request_duration_ms: number | null;
  startTime: string;
  endTime: string | null;
  api_key: string | null;
  team_id: string | null;
  end_user: string | null;
  metadata: any;
}

const MCPServerLogs: React.FC<MCPServerLogsProps> = ({
  serverId,
  accessToken,
}) => {
  const [logs, setLogs] = useState<MCPLogEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(25);
  const [timeRange, setTimeRange] = useState("7d");

  const { startDate, endDate } = useMemo(() => {
    const end = moment().utc().format("YYYY-MM-DD HH:mm:ss");
    let start: string;
    switch (timeRange) {
      case "1h":
        start = moment().utc().subtract(1, "hours").format("YYYY-MM-DD HH:mm:ss");
        break;
      case "24h":
        start = moment().utc().subtract(24, "hours").format("YYYY-MM-DD HH:mm:ss");
        break;
      case "7d":
        start = moment().utc().subtract(7, "days").format("YYYY-MM-DD HH:mm:ss");
        break;
      case "30d":
        start = moment().utc().subtract(30, "days").format("YYYY-MM-DD HH:mm:ss");
        break;
      default:
        start = moment().utc().subtract(7, "days").format("YYYY-MM-DD HH:mm:ss");
    }
    return { startDate: start, endDate: end };
  }, [timeRange]);

  useEffect(() => {
    const fetchLogs = async () => {
      if (!accessToken) return;
      setLoading(true);
      try {
        const result = await fetchMCPServerLogs(
          accessToken,
          serverId,
          startDate,
          endDate,
          page,
          pageSize,
        );
        setLogs(result.data || []);
        setTotal(result.total || 0);
      } catch (err) {
        console.error("Failed to fetch MCP server logs:", err);
        setLogs([]);
        setTotal(0);
      } finally {
        setLoading(false);
      }
    };
    fetchLogs();
  }, [accessToken, serverId, startDate, endDate, page, pageSize]);

  const extractToolName = (namespacedName: string | null) => {
    if (!namespacedName) return "—";
    const parts = namespacedName.split("/");
    return parts.length > 1 ? parts.slice(1).join("/") : namespacedName;
  };

  const columns = [
    {
      title: "Time",
      dataIndex: "startTime",
      key: "startTime",
      width: 180,
      render: (val: string) => (
        <Tooltip title={val}>
          <span className="text-xs">{moment(val).format("MMM DD HH:mm:ss")}</span>
        </Tooltip>
      ),
    },
    {
      title: "Tool",
      dataIndex: "mcp_namespaced_tool_name",
      key: "tool",
      render: (val: string | null) => (
        <span className="font-mono text-xs">{extractToolName(val)}</span>
      ),
    },
    {
      title: "Type",
      dataIndex: "call_type",
      key: "call_type",
      width: 130,
      render: (val: string) => (
        <Tag color={val === "call_mcp_tool" ? "blue" : "default"}>
          {val === "call_mcp_tool" ? "Tool Call" : val === "list_mcp_tools" ? "List Tools" : val}
        </Tag>
      ),
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      width: 90,
      render: (val: string | null) => {
        const isSuccess = !val || val === "success";
        return (
          <Tag color={isSuccess ? "green" : "red"}>
            {isSuccess ? "OK" : val}
          </Tag>
        );
      },
    },
    {
      title: "Latency",
      dataIndex: "request_duration_ms",
      key: "latency",
      width: 100,
      render: (val: number | null) => {
        if (val == null) return "—";
        if (val < 1000) return `${val}ms`;
        return `${(val / 1000).toFixed(1)}s`;
      },
    },
    {
      title: "Cost",
      dataIndex: "spend",
      key: "spend",
      width: 80,
      render: (val: number) => (val > 0 ? `$${val.toFixed(4)}` : "—"),
    },
    {
      title: "Caller",
      key: "caller",
      width: 150,
      render: (_: any, record: MCPLogEntry) => {
        const alias = record.metadata?.user_api_key_alias;
        return (
          <span className="text-xs truncate max-w-[140px] block">
            {alias || record.api_key?.slice(0, 12) || "—"}
          </span>
        );
      },
    },
  ];

  return (
    <Card>
      <div className="flex justify-between items-center mb-4">
        <Title>Invocation Logs</Title>
        <Select
          value={timeRange}
          onChange={setTimeRange}
          style={{ width: 140 }}
          options={[
            { label: "Last 1 hour", value: "1h" },
            { label: "Last 24 hours", value: "24h" },
            { label: "Last 7 days", value: "7d" },
            { label: "Last 30 days", value: "30d" },
          ]}
        />
      </div>

      {loading ? (
        <div className="flex justify-center py-8">
          <Spin />
        </div>
      ) : logs.length === 0 ? (
        <Empty
          description={
            <Text>No MCP invocation logs found for this server in the selected time range.</Text>
          }
        />
      ) : (
        <>
          <Table
            dataSource={logs}
            columns={columns}
            rowKey="request_id"
            pagination={false}
            size="small"
            scroll={{ x: 900 }}
          />
          {total > pageSize && (
            <div className="flex justify-end mt-4">
              <Pagination
                current={page}
                pageSize={pageSize}
                total={total}
                onChange={(p) => setPage(p)}
                showSizeChanger={false}
                showTotal={(total) => `${total} logs`}
              />
            </div>
          )}
        </>
      )}
    </Card>
  );
};

export default MCPServerLogs;
