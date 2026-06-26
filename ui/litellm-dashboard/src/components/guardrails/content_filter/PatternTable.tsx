import React from "react";
import { Typography, Select, Table, Tag, Button } from "antd";
import { DeleteOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";

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
  const { t } = useTranslation();

  const columns = [
    {
      title: t("guardrails.patternTable.colType"),
      dataIndex: "type",
      key: "type",
      width: 100,
      render: (type: string) => (
        <Tag color={type === "prebuilt" ? "blue" : "green"}>
          {type === "prebuilt" ? t("guardrails.patternTable.typePrebuilt") : t("guardrails.patternTable.typeCustom")}
        </Tag>
      ),
    },
    {
      title: t("guardrails.patternTable.colPatternName"),
      dataIndex: "name",
      key: "name",
      render: (_: string, record: Pattern) => record.display_name || record.name,
    },
    {
      title: t("guardrails.patternTable.colRegexPattern"),
      dataIndex: "pattern",
      key: "pattern",
      render: (pattern: string) =>
        pattern ? (
          <Text code style={{ fontSize: 12 }}>
            {pattern.substring(0, 40)}...
          </Text>
        ) : (
          "-"
        ),
    },
    {
      title: t("guardrails.patternTable.colAction"),
      dataIndex: "action",
      key: "action",
      width: 150,
      render: (action: string, record: Pattern) => (
        <Select
          value={action}
          onChange={(value) => onActionChange(record.id, value as "BLOCK" | "MASK")}
          style={{ width: 120 }}
          size="small"
        >
          <Option value="BLOCK">{t("guardrails.patternTable.actionBlock")}</Option>
          <Option value="MASK">{t("guardrails.patternTable.actionMask")}</Option>
        </Select>
      ),
    },
    {
      title: "",
      key: "actions",
      width: 100,
      render: (_: any, record: Pattern) => (
        <Button type="text" danger size="small" icon={<DeleteOutlined />} onClick={() => onRemove(record.id)}>
          {t("common.delete")}
        </Button>
      ),
    },
  ];

  if (patterns.length === 0) {
    return (
      <div style={{ textAlign: "center", padding: "40px 0", color: "#999" }}>
        {t("guardrails.patternTable.noPatterns")}
      </div>
    );
  }

  return <Table dataSource={patterns} columns={columns} rowKey="id" pagination={false} size="small" />;
};

export default PatternTable;
