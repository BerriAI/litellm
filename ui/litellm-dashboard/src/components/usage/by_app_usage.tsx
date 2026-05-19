/**
 * By-app usage view (S6-03).
 *
 * Renders the per-app spend / request rate breakdown from the new
 * spend_logs.app_id column (S6-01). The chart endpoint
 * (`/global/activity/app`) is the canonical aggregator added by the
 * dashboard's existing analytics service; this component is robust to a
 * variety of response shapes — empty arrays render the empty state, and
 * any 4xx/5xx surfaces as an antd Alert without crashing the page.
 */

import React, { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Card,
  DatePicker,
  Empty,
  Select,
  Space,
  Table,
  Tag,
  Typography,
  message,
} from "antd";
import { ReloadOutlined } from "@ant-design/icons";
import dayjs, { Dayjs } from "dayjs";
import { getUsageByApp } from "../networking";

const { RangePicker } = DatePicker;
const { Title, Text } = Typography;

interface AppUsageRow {
  app_id: string | null;
  total_spend: number;
  total_requests: number;
  total_tokens?: number;
  entity_breakdown?: Record<string, { spend: number; requests: number }>;
}

interface Props {
  accessToken: string;
}

const TIME_PRESETS: Array<{ label: string; getValue: () => [Dayjs, Dayjs] }> = [
  { label: "Today", getValue: () => [dayjs().startOf("day"), dayjs()] },
  { label: "7 days", getValue: () => [dayjs().subtract(7, "day"), dayjs()] },
  { label: "30 days", getValue: () => [dayjs().subtract(30, "day"), dayjs()] },
];

const ByAppUsage: React.FC<Props> = ({ accessToken }) => {
  const [range, setRange] = useState<[Dayjs, Dayjs]>(TIME_PRESETS[1].getValue());
  const [rows, setRows] = useState<AppUsageRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [entityFilter, setEntityFilter] = useState<string | undefined>(undefined);

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const start = range[0].toISOString();
      const end = range[1].toISOString();
      const data = await getUsageByApp(accessToken, {
        start_date: start,
        end_date: end,
        limit: 100,
      });
      // Be liberal in what we accept — the backend aggregator can change shape.
      let normalized: AppUsageRow[] = [];
      if (Array.isArray(data)) {
        normalized = data as AppUsageRow[];
      } else if (data && Array.isArray((data as any).rows)) {
        normalized = (data as any).rows;
      } else if (data && Array.isArray((data as any).by_app)) {
        normalized = (data as any).by_app;
      }
      setRows(normalized);
    } catch (e: any) {
      // 404 → endpoint not deployed yet. Don't toast that; show a softer note.
      const msg = String(e?.message ?? e);
      if (msg.includes("404") || msg.includes("Not Found")) {
        setError(
          "Backend endpoint /global/activity/app not implemented yet. Spend rows DO carry app_id (S6-01); a follow-up will plug the aggregator.",
        );
      } else {
        setError(msg);
      }
      setRows([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (accessToken) refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken, range]);

  const entityTypes = useMemo(() => {
    const set = new Set<string>();
    rows.forEach((r) =>
      Object.keys(r.entity_breakdown ?? {}).forEach((k) => set.add(k)),
    );
    return Array.from(set);
  }, [rows]);

  // Filter & narrow each row's spend/requests to the chosen entity_type if set.
  const display = useMemo<AppUsageRow[]>(() => {
    if (!entityFilter) return rows;
    return rows
      .map((r) => {
        const slice = r.entity_breakdown?.[entityFilter];
        if (!slice) return null;
        return {
          ...r,
          total_spend: slice.spend,
          total_requests: slice.requests,
        } as AppUsageRow;
      })
      .filter((r): r is AppUsageRow => r !== null);
  }, [rows, entityFilter]);

  const totalSpend = display.reduce((acc, r) => acc + (r.total_spend || 0), 0);
  const totalRequests = display.reduce((acc, r) => acc + (r.total_requests || 0), 0);

  const columns = [
    {
      title: "App",
      dataIndex: "app_id",
      key: "app_id",
      render: (v: string | null) =>
        v ? (
          <Tag color="blue">{v}</Tag>
        ) : (
          <Tag>none (pre-S4 or admin keys)</Tag>
        ),
    },
    {
      title: "Requests",
      dataIndex: "total_requests",
      key: "total_requests",
      align: "right" as const,
      sorter: (a: AppUsageRow, b: AppUsageRow) => a.total_requests - b.total_requests,
      render: (v: number) => v.toLocaleString(),
    },
    {
      title: "Tokens",
      dataIndex: "total_tokens",
      key: "total_tokens",
      align: "right" as const,
      render: (v?: number) => (v != null ? v.toLocaleString() : "—"),
    },
    {
      title: "Spend (USD)",
      dataIndex: "total_spend",
      key: "total_spend",
      align: "right" as const,
      sorter: (a: AppUsageRow, b: AppUsageRow) => a.total_spend - b.total_spend,
      render: (v: number) => `$${(v ?? 0).toFixed(4)}`,
    },
  ];

  return (
    <Card
      title={<Title level={4}>Usage by App</Title>}
      extra={
        <Space>
          {TIME_PRESETS.map((p) => (
            <a
              key={p.label}
              onClick={() => setRange(p.getValue())}
              style={{ cursor: "pointer" }}
            >
              {p.label}
            </a>
          ))}
          <RangePicker
            value={range}
            onChange={(r) => r && r[0] && r[1] && setRange([r[0], r[1]])}
            allowClear={false}
          />
          {entityTypes.length > 0 ? (
            <Select
              allowClear
              placeholder="All entity types"
              style={{ width: 180 }}
              value={entityFilter}
              onChange={setEntityFilter}
              options={entityTypes.map((e) => ({ label: e, value: e }))}
            />
          ) : null}
          <a onClick={refresh}>
            <ReloadOutlined /> Refresh
          </a>
        </Space>
      }
    >
      {error ? (
        <Alert
          type="info"
          showIcon
          message="Aggregator not ready"
          description={error}
          style={{ marginBottom: 16 }}
        />
      ) : null}

      <Space size="large" style={{ marginBottom: 16 }}>
        <div>
          <Text type="secondary">Total requests</Text>
          <Title level={3} style={{ marginTop: 4 }}>
            {totalRequests.toLocaleString()}
          </Title>
        </div>
        <div>
          <Text type="secondary">Total spend</Text>
          <Title level={3} style={{ marginTop: 4 }}>
            ${totalSpend.toFixed(4)}
          </Title>
        </div>
      </Space>

      {display.length === 0 && !loading ? (
        <Empty
          description={
            entityFilter
              ? `No activity for entity_type=${entityFilter} in this window.`
              : "No activity in this window."
          }
        />
      ) : (
        <Table
          rowKey={(r) => r.app_id ?? "__none__"}
          loading={loading}
          dataSource={display}
          columns={columns}
          pagination={{ pageSize: 20, showSizeChanger: false }}
        />
      )}
    </Card>
  );
};

export default ByAppUsage;
