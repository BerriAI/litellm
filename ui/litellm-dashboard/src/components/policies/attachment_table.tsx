import React from "react";
import { Table, Tag, Button, Space, Tooltip, Typography } from "antd";
import { DeleteOutlined } from "@ant-design/icons";
import { PolicyAttachment } from "./types";

const { Text } = Typography;

interface AttachmentTableProps {
  attachments: PolicyAttachment[];
  isLoading: boolean;
  onDeleteClick: (attachmentId: string) => void;
  isAdmin: boolean;
}

const AttachmentTable: React.FC<AttachmentTableProps> = ({
  attachments,
  isLoading,
  onDeleteClick,
  isAdmin,
}) => {
  const columns = [
    {
      title: "Policy",
      dataIndex: "policy_name",
      key: "policy_name",
      render: (text: string) => <Tag color="blue">{text}</Tag>,
    },
    {
      title: "Scope",
      dataIndex: "scope",
      key: "scope",
      render: (scope: string | null) =>
        scope === "*" ? (
          <Tag color="gold">Global (*)</Tag>
        ) : scope ? (
          <Tag>{scope}</Tag>
        ) : (
          <Text type="secondary">-</Text>
        ),
    },
    {
      title: "Teams",
      dataIndex: "teams",
      key: "teams",
      render: (teams: string[]) => (
        <Space wrap>
          {teams && teams.length > 0 ? (
            teams.slice(0, 3).map((t) => (
              <Tag key={t} color="cyan">
                {t}
              </Tag>
            ))
          ) : (
            <Text type="secondary">-</Text>
          )}
          {teams && teams.length > 3 && (
            <Tooltip title={teams.slice(3).join(", ")}>
              <Tag>+{teams.length - 3} more</Tag>
            </Tooltip>
          )}
        </Space>
      ),
    },
    {
      title: "Keys",
      dataIndex: "keys",
      key: "keys",
      render: (keys: string[]) => (
        <Space wrap>
          {keys && keys.length > 0 ? (
            keys.slice(0, 3).map((k) => (
              <Tag key={k} color="purple">
                {k}
              </Tag>
            ))
          ) : (
            <Text type="secondary">-</Text>
          )}
          {keys && keys.length > 3 && (
            <Tooltip title={keys.slice(3).join(", ")}>
              <Tag>+{keys.length - 3} more</Tag>
            </Tooltip>
          )}
        </Space>
      ),
    },
    {
      title: "Models",
      dataIndex: "models",
      key: "models",
      render: (models: string[]) => (
        <Space wrap>
          {models && models.length > 0 ? (
            models.slice(0, 3).map((m) => (
              <Tag key={m} color="green">
                {m}
              </Tag>
            ))
          ) : (
            <Text type="secondary">-</Text>
          )}
          {models && models.length > 3 && (
            <Tooltip title={models.slice(3).join(", ")}>
              <Tag>+{models.length - 3} more</Tag>
            </Tooltip>
          )}
        </Space>
      ),
    },
    {
      title: "Created At",
      dataIndex: "created_at",
      key: "created_at",
      render: (date: string) =>
        date ? new Date(date).toLocaleDateString() : "-",
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: any, record: PolicyAttachment) => (
        <Space>
          {isAdmin && (
            <Tooltip title="Delete">
              <Button
                type="text"
                danger
                icon={<DeleteOutlined />}
                onClick={() => onDeleteClick(record.attachment_id)}
              />
            </Tooltip>
          )}
        </Space>
      ),
    },
  ];

  return (
    <Table
      columns={columns}
      dataSource={attachments}
      rowKey="attachment_id"
      loading={isLoading}
      pagination={{
        pageSize: 10,
        showSizeChanger: true,
        showTotal: (total, range) =>
          `${range[0]}-${range[1]} of ${total} attachments`,
      }}
    />
  );
};

export default AttachmentTable;
