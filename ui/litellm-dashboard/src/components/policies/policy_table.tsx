import React from "react";
import { Table, Tag, Button, Space, Tooltip, Typography } from "antd";
import { DeleteOutlined, EditOutlined, EyeOutlined } from "@ant-design/icons";
import { Policy } from "./types";

const { Text } = Typography;

interface PolicyTableProps {
  policies: Policy[];
  isLoading: boolean;
  onDeleteClick: (policyId: string, policyName: string) => void;
  onEditClick: (policy: Policy) => void;
  onViewClick: (policyId: string) => void;
  isAdmin: boolean;
}

const PolicyTable: React.FC<PolicyTableProps> = ({
  policies,
  isLoading,
  onDeleteClick,
  onEditClick,
  onViewClick,
  isAdmin,
}) => {
  const columns = [
    {
      title: "Policy Name",
      dataIndex: "policy_name",
      key: "policy_name",
      render: (text: string, record: Policy) => (
        <Button type="link" onClick={() => onViewClick(record.policy_id)}>
          {text}
        </Button>
      ),
    },
    {
      title: "Description",
      dataIndex: "description",
      key: "description",
      render: (text: string | null) => text || <Text type="secondary">-</Text>,
    },
    {
      title: "Inherits From",
      dataIndex: "inherit",
      key: "inherit",
      render: (text: string | null) =>
        text ? <Tag color="blue">{text}</Tag> : <Text type="secondary">-</Text>,
    },
    {
      title: "Guardrails (Add)",
      dataIndex: "guardrails_add",
      key: "guardrails_add",
      render: (guardrails: string[]) => (
        <Space wrap>
          {guardrails && guardrails.length > 0 ? (
            guardrails.slice(0, 3).map((g) => (
              <Tag key={g} color="green">
                {g}
              </Tag>
            ))
          ) : (
            <Text type="secondary">-</Text>
          )}
          {guardrails && guardrails.length > 3 && (
            <Tooltip title={guardrails.slice(3).join(", ")}>
              <Tag>+{guardrails.length - 3} more</Tag>
            </Tooltip>
          )}
        </Space>
      ),
    },
    {
      title: "Guardrails (Remove)",
      dataIndex: "guardrails_remove",
      key: "guardrails_remove",
      render: (guardrails: string[]) => (
        <Space wrap>
          {guardrails && guardrails.length > 0 ? (
            guardrails.slice(0, 3).map((g) => (
              <Tag key={g} color="red">
                {g}
              </Tag>
            ))
          ) : (
            <Text type="secondary">-</Text>
          )}
          {guardrails && guardrails.length > 3 && (
            <Tooltip title={guardrails.slice(3).join(", ")}>
              <Tag>+{guardrails.length - 3} more</Tag>
            </Tooltip>
          )}
        </Space>
      ),
    },
    {
      title: "Model Condition",
      dataIndex: "condition",
      key: "condition",
      render: (condition: { model?: string } | null) =>
        condition?.model ? (
          <Tag color="purple">{condition.model}</Tag>
        ) : (
          <Text type="secondary">-</Text>
        ),
    },
    {
      title: "Actions",
      key: "actions",
      render: (_: any, record: Policy) => (
        <Space>
          <Tooltip title="View">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => onViewClick(record.policy_id)}
            />
          </Tooltip>
          {isAdmin && (
            <>
              <Tooltip title="Edit">
                <Button
                  type="text"
                  icon={<EditOutlined />}
                  onClick={() => onEditClick(record)}
                />
              </Tooltip>
              <Tooltip title="Delete">
                <Button
                  type="text"
                  danger
                  icon={<DeleteOutlined />}
                  onClick={() =>
                    onDeleteClick(record.policy_id, record.policy_name)
                  }
                />
              </Tooltip>
            </>
          )}
        </Space>
      ),
    },
  ];

  return (
    <Table
      columns={columns}
      dataSource={policies}
      rowKey="policy_id"
      loading={isLoading}
      pagination={{
        pageSize: 10,
        showSizeChanger: true,
        showTotal: (total, range) =>
          `${range[0]}-${range[1]} of ${total} policies`,
      }}
    />
  );
};

export default PolicyTable;
