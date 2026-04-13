import React from "react";
import { Typography, Select, Table, Tag, Button } from "antd";
import { DeleteOutlined } from "@ant-design/icons";

const { Text } = Typography;
const { Option } = Select;

interface ContentCategory {
  id: string;
  category: string;
  display_name: string;
  action: "BLOCK" | "MASK";
  severity_threshold: "high" | "medium" | "low";
}

interface CategoryTableProps {
  categories: ContentCategory[];
  onActionChange?: (id: string, action: "BLOCK" | "MASK") => void;
  onSeverityChange?: (id: string, severity: "high" | "medium" | "low") => void;
  onRemove?: (id: string) => void;
  readOnly?: boolean;
}

const CategoryTable: React.FC<CategoryTableProps> = ({
  categories,
  onActionChange,
  onSeverityChange,
  onRemove,
  readOnly = false,
}) => {
  const columns = [
    {
      title: "Category",
      dataIndex: "display_name",
      key: "display_name",
      render: (displayName: string, record: ContentCategory) => (
        <div>
          <Text strong>{displayName}</Text>
          {displayName !== record.category && (
            <div>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {record.category}
              </Text>
            </div>
          )}
        </div>
      ),
    },
    {
      title: "Severity Threshold",
      dataIndex: "severity_threshold",
      key: "severity_threshold",
      width: 180,
      render: (severity: string, record: ContentCategory) => {
        if (readOnly) {
          const colorMap = {
            high: "red",
            medium: "orange",
            low: "yellow",
          } as const;
          return (
            <Tag color={colorMap[severity as keyof typeof colorMap]}>
              {severity.toUpperCase()}
            </Tag>
          );
        }
        return (
          <Select
            value={severity}
            onChange={(value) => onSeverityChange?.(record.id, value as "high" | "medium" | "low")}
            style={{ width: 150 }}
            size="small"
          >
            <Option value="high">High</Option>
            <Option value="medium">Medium</Option>
            <Option value="low">Low</Option>
          </Select>
        );
      },
    },
    {
      title: "Action",
      dataIndex: "action",
      key: "action",
      width: 150,
      render: (action: string, record: ContentCategory) => {
        if (readOnly) {
          return (
            <Tag color={action === "BLOCK" ? "red" : "blue"}>
              {action}
            </Tag>
          );
        }
        return (
          <Select
            value={action}
            onChange={(value) => onActionChange?.(record.id, value as "BLOCK" | "MASK")}
            style={{ width: 120 }}
            size="small"
          >
            <Option value="BLOCK">Block</Option>
            <Option value="MASK">Mask</Option>
          </Select>
        );
      },
    },
  ];

  if (!readOnly) {
    columns.push({
      title: "",
      key: "actions",
      width: 100,
      render: (_: any, record: ContentCategory) => (
        <Button
          type="text"
          danger
          size="small"
          icon={<DeleteOutlined />}
          onClick={() => onRemove?.(record.id)}
        >
          Delete
        </Button>
      ),
    } as any);
  }

  if (categories.length === 0) {
    return (
      <div style={{ textAlign: "center", padding: "40px 0", color: "#999" }}>
        No categories configured.
      </div>
    );
  }

  return (
    <Table
      dataSource={categories}
      columns={columns}
      rowKey="id"
      pagination={false}
      size="small"
    />
  );
};

export default CategoryTable;
