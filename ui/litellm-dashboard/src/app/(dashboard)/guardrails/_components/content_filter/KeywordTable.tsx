import { DeleteOutlined } from "@ant-design/icons";
import type { ColumnDef } from "@tanstack/react-table";
import { Button, Select } from "antd";
import React from "react";
import { DataTable } from "@/components/shared/DataTable";

const { Option } = Select;

interface BlockedWord {
  id: string;
  keyword: string;
  action: "BLOCK" | "MASK";
  description?: string;
}

interface KeywordTableProps {
  keywords: BlockedWord[];
  onActionChange: (id: string, field: string, value: any) => void;
  onRemove: (id: string) => void;
}

const KeywordTable: React.FC<KeywordTableProps> = ({ keywords, onActionChange, onRemove }) => {
  const columns: ColumnDef<BlockedWord>[] = [
    {
      header: "Keyword",
      accessorKey: "keyword",
    },
    {
      header: "Action",
      accessorKey: "action",
      size: 150,
      cell: ({ row }) => (
        <Select
          value={row.original.action}
          onChange={(value) => onActionChange(row.original.id, "action", value)}
          style={{ width: 120 }}
          size="small"
        >
          <Option value="BLOCK">Block</Option>
          <Option value="MASK">Mask</Option>
        </Select>
      ),
    },
    {
      header: "Description",
      accessorKey: "description",
      cell: ({ row }) => row.original.description || "-",
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

  if (keywords.length === 0) {
    return <div style={{ textAlign: "center", padding: "40px 0", color: "#999" }}>No keywords added.</div>;
  }

  return <DataTable data={keywords} columns={columns} getRowId={(row) => row.id} size="compact" />;
};

export default KeywordTable;
