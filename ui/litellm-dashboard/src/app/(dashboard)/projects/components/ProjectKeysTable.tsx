import { KeyResponse } from "@/components/key_team_helpers/key_list";
import { Empty, Table, Tooltip } from "antd";
import type { ColumnsType } from "antd/es/table";
import type { SpinProps } from "antd";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import { useMemo } from "react";
import DefaultProxyAdminTag from "@/components/common_components/DefaultProxyAdminTag";

interface ProjectKeysTableProps {
  keys: KeyResponse[];
  loading?: boolean | SpinProps;
}

const getProjectKeysTableColumns = (t: TFunction): ColumnsType<KeyResponse> => [
  {
    title: t("projects.projectKeysTable.colKeyName"),
    dataIndex: "key_alias",
    key: "key_alias",
    render: (alias: string | null) => alias || "—",
  },
  {
    title: t("projects.projectKeysTable.colOwner"),
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
    title: t("common.createdAt"),
    dataIndex: "created_at",
    key: "created_at",
    render: (date: string) => (date ? new Date(date).toLocaleDateString() : "—"),
  },
  {
    title: t("projects.projectKeysTable.colLastActive"),
    dataIndex: "last_active",
    key: "last_active",
    render: (date: string | null) => (date ? new Date(date).toLocaleDateString() : t("common.never")),
  },
];

export function ProjectKeysTable({ keys, loading }: ProjectKeysTableProps) {
  const { t } = useTranslation();
  const columns = useMemo(() => getProjectKeysTableColumns(t), [t]);

  return (
    <Table
      columns={columns}
      dataSource={keys}
      rowKey="token"
      loading={loading}
      pagination={false}
      size="small"
      locale={{
        emptyText: (
          <Empty description={t("projects.projectKeysTable.noKeysFound")} image={Empty.PRESENTED_IMAGE_SIMPLE} />
        ),
      }}
    />
  );
}
