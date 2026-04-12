import { KeyResponse } from "@/components/key_team_helpers/key_list";
import { Empty, Table, Tooltip } from "antd";
import type { ColumnsType } from "antd/es/table";
import type { SpinProps } from "antd";
import DefaultProxyAdminTag from "../common_components/DefaultProxyAdminTag";

interface ProjectKeysTableProps {
  keys: KeyResponse[];
  loading?: boolean | SpinProps;
}

const columns: ColumnsType<KeyResponse> = [
  {
    title: "Key Name",
    dataIndex: "key_alias",
    key: "key_alias",
    render: (alias: string | null) => alias || "—",
  },
  {
    title: "Owner",
    key: "owner",
    render: (_: unknown, record: KeyResponse) => {
      const email = record.user?.user_email ?? record.user_id ?? null;
      if (!email) return "—";
      return (
        <Tooltip title={email}>
          <DefaultProxyAdminTag userId={email} />
        </Tooltip>
      );
    },
  },
  {
    title: "Created",
    dataIndex: "created_at",
    key: "created_at",
    render: (date: string) => (date ? new Date(date).toLocaleDateString() : "—"),
  },
  {
    title: "Last Active",
    dataIndex: "last_active",
    key: "last_active",
    render: (date: string | null) => (date ? new Date(date).toLocaleDateString() : "Never"),
  },
];

export function ProjectKeysTable({ keys, loading }: ProjectKeysTableProps) {
  return (
    <Table
      columns={columns}
      dataSource={keys}
      rowKey="token"
      loading={loading}
      pagination={false}
      size="small"
      locale={{ emptyText: <Empty description="No keys found" image={Empty.PRESENTED_IMAGE_SIMPLE} /> }}
    />
  );
}
