"use client";

import Link from "next/link";
import { Card, Table, Tag, Typography, Empty } from "antd";
import type { ColumnsType } from "antd/es/table";
import { relativeOrAbsolute } from "@/app/(dashboard)/agents/components/_dayjs";
import type { CloudAgent } from "@/types/cloud-agents";

const { Text } = Typography;

interface AgentListProps {
  agents: CloudAgent[];
  loading: boolean;
}

export default function AgentList({ agents, loading }: AgentListProps) {
  const columns: ColumnsType<CloudAgent> = [
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
      render: (_value: unknown, record: CloudAgent) => (
        <Link href={`/agents/${record.agent_id}`} data-testid={`agent-link-${record.agent_id}`}>
          <Text strong>{record.name}</Text>
        </Link>
      ),
    },
    {
      title: "Model",
      dataIndex: "model",
      key: "model",
      render: (value: string) => <Tag color="blue">{value}</Tag>,
    },
    {
      title: "Sessions",
      dataIndex: "session_count",
      key: "session_count",
      width: 100,
    },
    {
      title: "Last activity",
      dataIndex: "last_activity_at",
      key: "last_activity_at",
      render: (value: string | null) => relativeOrAbsolute(value),
    },
  ];

  return (
    <Card>
      <Table<CloudAgent>
        rowKey="agent_id"
        dataSource={agents}
        columns={columns}
        loading={loading}
        locale={{
          emptyText: <Empty description="No cloud agents yet — create your first one." />,
        }}
        data-testid="agents-table"
      />
    </Card>
  );
}
