import React from "react";
import { Typography, Select, Tag, Button } from "antd";
import { DeleteOutlined } from "@ant-design/icons";
import type { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "@/components/shared/DataTable";

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
  const columns: ColumnDef<ContentCategory>[] = [
    {
      header: "Category",
      accessorKey: "display_name",
      cell: ({ row }) => {
        const { category, display_name: displayName } = row.original;
        return (
          <div>
            <Text strong>{displayName}</Text>
            {displayName !== category && (
              <div>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {category}
                </Text>
              </div>
            )}
          </div>
        );
      },
    },
    {
      header: "Severity Threshold",
      accessorKey: "severity_threshold",
      size: 180,
      cell: ({ row }) => {
        const { id, severity_threshold: severity } = row.original;
        if (readOnly) {
          const colorMap = {
            high: "red",
            medium: "orange",
            low: "yellow",
          } as const;
          return <Tag color={colorMap[severity as keyof typeof colorMap]}>{severity.toUpperCase()}</Tag>;
        }
        return (
          <Select
            value={severity}
            onChange={(value) => onSeverityChange?.(id, value as "high" | "medium" | "low")}
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
      header: "Action",
      accessorKey: "action",
      size: 150,
      cell: ({ row }) => {
        const { action, id } = row.original;
        if (readOnly) {
          return <Tag color={action === "BLOCK" ? "red" : "blue"}>{action}</Tag>;
        }
        return (
          <Select
            value={action}
            onChange={(value) => onActionChange?.(id, value as "BLOCK" | "MASK")}
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
      header: "",
      id: "actions",
      size: 100,
      cell: ({ row }) => (
        <Button type="text" danger size="small" icon={<DeleteOutlined />} onClick={() => onRemove?.(row.original.id)}>
          Delete
        </Button>
      ),
    });
  }

  if (categories.length === 0) {
    return <div style={{ textAlign: "center", padding: "40px 0", color: "#999" }}>No categories configured.</div>;
  }

  return <DataTable data={categories} columns={columns} getRowId={(row) => row.id} size="compact" />;
};

export default CategoryTable;
