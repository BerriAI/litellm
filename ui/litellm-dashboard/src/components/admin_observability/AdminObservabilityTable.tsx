"use client";

import { useMemo, useState } from "react";
import { Button, Card, DatePicker, Input, Select, Space, Spin, Table, Tag } from "antd";
import type { ColumnsType, TablePaginationConfig } from "antd/es/table";
import type { FilterValue, SorterResult } from "antd/es/table/interface";
import dayjs from "dayjs";
import type { AdminObservabilityFilters, AdminObservabilityRow } from "./types";
import { useAdminObservability } from "./useAdminObservability";
import { uiSpendLogDetailsCall } from "@/components/networking";
import { getSpendString } from "@/utils/dataUtils";
import { parseToolsFromLog } from "@/components/view_logs/ToolsSection/utils";
import { createAdminObservabilityColumns } from "./columns";
import { TimeCell } from "@/components/view_logs/time_cell";

const { RangePicker } = DatePicker;

interface AdminObservabilityTableProps {
  accessToken: string | null;
}

const DATE_FORMAT = "YYYY-MM-DD HH:mm:ss";

function formatJson(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      return value;
    }
  }
  return JSON.stringify(value, null, 2);
}

const detailColumns: ColumnsType<AdminObservabilityRow> = [
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
    width: 140,
    render: (value: string) => <span className="truncate block">{value || "-"}</span>,
  },
  {
    title: "Model",
    dataIndex: "model",
    key: "model",
    width: 160,
    render: (value: string) => <span className="truncate block">{value || "-"}</span>,
  },
  {
    title: "Input",
    dataIndex: "prompt_tokens",
    key: "prompt_tokens",
    width: 80,
    align: "right",
    render: (value: number | undefined) => value ?? 0,
  },
  {
    title: "Output",
    dataIndex: "completion_tokens",
    key: "completion_tokens",
    width: 80,
    align: "right",
    render: (value: number | undefined) => value ?? 0,
  },
  {
    title: "Cost",
    dataIndex: "spend",
    key: "spend",
    width: 100,
    render: (value: number | undefined) => getSpendString(value ?? 0),
  },
];

export default function AdminObservabilityTable({ accessToken }: AdminObservabilityTableProps) {
  const [dates, setDates] = useState<[dayjs.Dayjs, dayjs.Dayjs]>([dayjs().subtract(7, "days"), dayjs()]);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [filters, setFilters] = useState<AdminObservabilityFilters>({});
  const [appliedFilters, setAppliedFilters] = useState<AdminObservabilityFilters>({});
  const [sortBy, setSortBy] = useState("startTime");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
  const [detailCache, setDetailCache] = useState<Record<string, AdminObservabilityRow | null>>({});
  const [loadingDetails, setLoadingDetails] = useState<Record<string, boolean>>({});

  const startDate = useMemo(() => dates[0].format(DATE_FORMAT), [dates]);
  const endDate = useMemo(() => dates[1].format(DATE_FORMAT), [dates]);

  const { data, isLoading, error } = useAdminObservability({
    accessToken,
    startDate,
    endDate,
    page,
    pageSize,
    filters: appliedFilters,
    sortBy,
    sortOrder,
  });

  const logs = useMemo(() => data?.data ?? [], [data]);
  const total = useMemo(() => data?.total ?? 0, [data]);

  const handleSearch = () => {
    setAppliedFilters(filters);
    setPage(1);
  };

  const handleReset = () => {
    const reset: AdminObservabilityFilters = {};
    setFilters(reset);
    setAppliedFilters(reset);
    setDates([dayjs().subtract(7, "days"), dayjs()]);
    setSortBy("startTime");
    setSortOrder("desc");
    setPage(1);
  };

  const handleTableChange = (
    pagination: TablePaginationConfig,
    _filters: Record<string, FilterValue | null>,
    sorter: SorterResult<AdminObservabilityRow> | SorterResult<AdminObservabilityRow>[],
  ) => {
    if (pagination.current && pagination.current !== page) {
      setPage(pagination.current);
    }
    if (pagination.pageSize && pagination.pageSize !== pageSize) {
      setPageSize(pagination.pageSize);
      setPage(1);
    }
    const singleSorter = Array.isArray(sorter) ? sorter[0] : sorter;
    if (singleSorter?.order) {
      setSortBy(String(singleSorter.field));
      setSortOrder(singleSorter.order === "ascend" ? "asc" : "desc");
    } else {
      setSortBy("startTime");
      setSortOrder("desc");
    }
  };

  const fetchDetail = async (requestId: string) => {
    if (!accessToken || detailCache[requestId] !== undefined) return;
    setLoadingDetails((prev) => ({ ...prev, [requestId]: true }));
    try {
      const detail = await uiSpendLogDetailsCall(accessToken, requestId, startDate);
      setDetailCache((prev) => ({ ...prev, [requestId]: detail ?? null }));
    } catch {
      setDetailCache((prev) => ({ ...prev, [requestId]: null }));
    } finally {
      setLoadingDetails((prev) => ({ ...prev, [requestId]: false }));
    }
  };

  const expandedRowRender = (row: AdminObservabilityRow) => {
    const detail = detailCache[row.request_id];
    if (loadingDetails[row.request_id]) {
      return (
        <div className="p-6 flex justify-center">
          <Spin />
        </div>
      );
    }
    if (detail === null) {
      return <div className="p-6 text-red-500">Failed to load request details</div>;
    }
    if (!detail) {
      void fetchDetail(row.request_id);
      return (
        <div className="p-6 flex justify-center">
          <Spin />
        </div>
      );
    }

    const parsedTools = parseToolsFromLog(detail).filter((t) => t.called);
    const metadataToolCalls = detail.metadata?.mcp_tool_call_metadata?.tool_calls ?? [];
    const allToolNames = new Set<string>();
    parsedTools.forEach((t) => allToolNames.add(t.name));
    if (Array.isArray(metadataToolCalls)) {
      metadataToolCalls.forEach((tc: unknown) => {
        if (tc !== null && typeof tc === "object") {
          const record = tc as Record<string, unknown>;
          const name = (record.function as Record<string, unknown>)?.name || record.name;
          if (typeof name === "string" && name) allToolNames.add(name);
        }
      });
    }
    const mcpName = detail.mcp_namespaced_tool_name || detail.metadata?.mcp_tool_call_metadata?.namespaced_tool_name;
    if (mcpName) allToolNames.add(mcpName);
    const toolNames = Array.from(allToolNames);

    const hasRequest = detail.messages && Object.keys(detail.messages).length > 0;
    const hasResponse = detail.response && Object.keys(detail.response).length > 0;

    return (
      <div className="p-4 bg-gray-50 space-y-4">
        {toolNames.length > 0 && (
          <div>
            <h4 className="text-sm font-medium mb-2">Tool Calls ({toolNames.length})</h4>
            <Space wrap size="small">
              {toolNames.map((name) => (
                <Tag key={name} color="blue">
                  {name}
                </Tag>
              ))}
            </Space>
          </div>
        )}

        <div className="overflow-hidden rounded-lg border">
          <Table
            columns={detailColumns}
            dataSource={[detail]}
            rowKey="request_id"
            pagination={false}
            size="small"
            bordered
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="space-y-1">
            <h4 className="text-sm font-medium">Request</h4>
            <pre className="bg-white p-3 rounded border text-xs overflow-auto max-h-96">
              {hasRequest ? formatJson(detail.messages) : "Request payload not stored for this log row."}
            </pre>
          </div>
          <div className="space-y-1">
            <h4 className="text-sm font-medium">Response</h4>
            <pre className="bg-white p-3 rounded border text-xs overflow-auto max-h-96">
              {hasResponse ? formatJson(detail.response) : "Response payload not stored for this log row."}
            </pre>
          </div>
        </div>
      </div>
    );
  };

  const columns = useMemo(() => createAdminObservabilityColumns(), []);

  if (!accessToken) {
    return (
      <div className="flex justify-center mt-20">
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="mb-4">
        <h1 className="text-2xl font-bold">Admin Observability</h1>
        <p className="text-gray-500 text-sm">View all users request logs, token usage, and tool calls</p>
      </div>

      <Card className="mb-4">
        <Space wrap align="start">
          <RangePicker
            showTime={{ format: "HH:mm" }}
            format="YYYY-MM-DD HH:mm"
            value={dates}
            onChange={(vals) => {
              if (vals && vals[0] && vals[1]) {
                setDates([vals[0], vals[1]]);
              }
            }}
          />
          <Input
            placeholder="User ID"
            value={filters.user_id}
            onChange={(e) => setFilters((f) => ({ ...f, user_id: e.target.value }))}
            style={{ width: 160 }}
          />
          <Input
            placeholder="Model"
            value={filters.model}
            onChange={(e) => setFilters((f) => ({ ...f, model: e.target.value }))}
            style={{ width: 160 }}
          />
          <Select
            placeholder="Status"
            allowClear
            value={filters.status_filter}
            onChange={(value) => setFilters((f) => ({ ...f, status_filter: value }))}
            style={{ width: 120 }}
            options={[
              { value: "success", label: "Success" },
              { value: "failure", label: "Failure" },
            ]}
          />
          <Input
            placeholder="Request ID"
            value={filters.request_id}
            onChange={(e) => setFilters((f) => ({ ...f, request_id: e.target.value }))}
            style={{ width: 200 }}
          />
          <Button type="primary" onClick={handleSearch}>
            Search
          </Button>
          <Button onClick={handleReset}>Reset</Button>
        </Space>
      </Card>

      {error && <div className="text-red-500 mb-4">Failed to load logs: {(error as Error).message}</div>}

      <Table
        columns={columns}
        dataSource={logs}
        rowKey="request_id"
        loading={isLoading}
        pagination={{
          current: page,
          pageSize,
          total,
          showSizeChanger: true,
          pageSizeOptions: [10, 25, 50, 100],
          showTotal: (t, range) => `${range[0]}-${range[1]} of ${t} logs`,
        }}
        expandable={{
          expandedRowRender,
          rowExpandable: () => true,
          expandRowByClick: false,
        }}
        onChange={handleTableChange}
        scroll={{ x: "max-content" }}
      />
    </div>
  );
}
