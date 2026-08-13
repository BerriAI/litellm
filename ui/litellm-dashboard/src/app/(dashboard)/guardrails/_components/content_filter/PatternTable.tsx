import React from "react";
import { Typography, Select, Tag, Button } from "antd";
import { DeleteOutlined } from "@ant-design/icons";
import type { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "@/components/shared/DataTable";

const { Text } = Typography;
const { Option } = Select;

interface Pattern {
  id: string;
  type: "prebuilt" | "custom";
  name: string;
  display_name?: string;
  pattern?: string;
  action: "BLOCK" | "MASK";
}

interface PatternTableProps {
  patterns: Pattern[];
  onActionChange: (id: string, action: "BLOCK" | "MASK") => void;
  onRemove: (id: string) => void;
}

const PatternTable: React.FC<PatternTableProps> = ({ patterns, onActionChange, onRemove }) => {
  const columns: ColumnDef<Pattern>[] = [
    {
      header: "Type",
      accessorKey: "type",
      size: 100,
      cell: ({ row }) => (
        <Tag color={row.original.type === "prebuilt" ? "blue" : "green"}>
          {row.original.type === "prebuilt" ? "Prebuilt" : "Custom"}
        </Tag>
      ),
    },
    {
      header: "Pattern name",
      accessorKey: "name",
      cell: ({ row }) => row.original.display_name || row.original.name,
    },
    {
      header: "Regex pattern",
      accessorKey: "pattern",
      cell: ({ row }) =>
        row.original.pattern ? (
          <Text code style={{ fontSize: 12 }}>
            {row.original.pattern.substring(0, 40)}...
          </Text>
        ) : (
          "-"
        ),
    },
    {
      header: "Action",
      accessorKey: "action",
      size: 150,
      cell: ({ row }) => (
        <Select
          value={row.original.action}
          onChange={(value) => onActionChange(row.original.id, value as "BLOCK" | "MASK")}
          style={{ width: 120 }}
          size="small"
        >
          <Option value="BLOCK">Block</Option>
          <Option value="MASK">Mask</Option>
        </Select>
      ),
    },
    {
      header: "",
      id: "actions",
      size: 100,
      cell: ({ row }) => (
        <Button type="text" danger size="small" icon={<DeleteOutlined />} onClick={() => onRemove(row.original.id)}>
          Delete
        </Button>
      ),
    },
  ];

  if (patterns.length === 0) {
    return <div style={{ textAlign: "center", padding: "40px 0", color: "#999" }}>No patterns added.</div>;
  }

  return <DataTable data={patterns} columns={columns} getRowId={(row) => row.id} size="compact" />;
};

export default PatternTable;
