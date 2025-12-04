import React from "react";
import { Typography, Select, Table } from "antd";
import { DeleteOutlined } from "@ant-design/icons";
import { Button } from "@tremor/react";

const { Text } = Typography;
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

const KeywordTable: React.FC<KeywordTableProps> = ({
  keywords,
  onActionChange,
  onRemove,
}) => {
  const columns = [
    {
      title: "Keyword",
      dataIndex: "keyword",
      key: "keyword",
    },
    {
      title: "Action",
      dataIndex: "action",
      key: "action",
      width: 150,
      render: (action: string, record: BlockedWord) => (
        <Select
          value={action}
          onChange={(value) => onActionChange(record.id, "action", value)}
          style={{ width: 120 }}
          size="small"
        >
          <Option value="BLOCK">Block</Option>
          <Option value="MASK">Mask</Option>
        </Select>
      ),
    },
    {
      title: "Description",
      dataIndex: "description",
      key: "description",
      render: (desc: string) => desc || "-",
    },
    {
      title: "",
      key: "actions",
      width: 100,
      render: (_: any, record: BlockedWord) => (
        <Button
          type="button"
          variant="light"
          color="red"
          size="xs"
          icon={DeleteOutlined}
          onClick={() => onRemove(record.id)}
        >
          Delete
        </Button>
      ),
    },
  ];

  if (keywords.length === 0) {
    return (
      <div style={{ textAlign: "center", padding: "40px 0", color: "#999" }}>
        No keywords added.
      </div>
    );
  }

  return (
    <Table
      dataSource={keywords}
      columns={columns}
      rowKey="id"
      pagination={false}
      size="small"
    />
  );
};

export default KeywordTable;

