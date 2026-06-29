import { DeleteOutlined } from "@ant-design/icons";
import { Button, Select, Table, Typography } from "antd";
import React from "react";
import { t } from "@/i18n";

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

const KeywordTable: React.FC<KeywordTableProps> = ({ keywords, onActionChange, onRemove }) => {
  const columns = [
    {
      title: t("keys.key"),
      dataIndex: "keyword",
      key: "keyword",
    },
    {
      title: t("common.actions"),
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
      title: t("models.base_model"),
      dataIndex: "description",
      key: "description",
      render: (desc: string) => desc || "-",
    },
    {
      title: "",
      key: "actions",
      width: 100,
      render: (_: any, record: BlockedWord) => (
        <Button type="text" danger size="small" icon={<DeleteOutlined />} onClick={() => onRemove(record.id)}>
          Delete
        </Button>
      ),
    },
  ];

  if (keywords.length === 0) {
    return <div style={{ textAlign: "center", padding: "40px 0", color: "#999" }}>No keywords added.</div>;
  }

  return <Table dataSource={keywords} columns={columns} rowKey="id" pagination={false} size="small" />;
};

export default KeywordTable;
