/**
 * Formatted view of tool definition with parameters table and call data
 */

import { Typography, Table } from "antd";
import { useTranslation } from "react-i18next";
import { ParsedTool, ParameterRow } from "./types";

const { Text } = Typography;

interface FormattedToolViewProps {
  tool: ParsedTool;
}

export function FormattedToolView({ tool }: FormattedToolViewProps) {
  const { t } = useTranslation();
  // Parse parameters for table display
  const parameterRows: ParameterRow[] = Object.entries(tool.parameters?.properties || {}).map(
    ([name, schema]: [string, any]) => ({
      key: name,
      name: name,
      type: schema.type || "any",
      description: schema.description || "-",
      required: tool.parameters?.required?.includes(name) || false,
    }),
  );

  const columns = [
    {
      title: t("viewLogs.formattedToolView.colParameter"),
      dataIndex: "name",
      key: "name",
      render: (name: string, record: ParameterRow) => (
        <Text code>
          {name}
          {record.required && <Text type="danger">*</Text>}
        </Text>
      ),
    },
    {
      title: t("viewLogs.formattedToolView.colType"),
      dataIndex: "type",
      key: "type",
      render: (type: string) => (
        <Text code style={{ color: "#1890ff" }}>
          {type}
        </Text>
      ),
    },
    {
      title: t("viewLogs.formattedToolView.colDescription"),
      dataIndex: "description",
      key: "description",
      render: (desc: string) => <Text type="secondary">{desc}</Text>,
    },
  ];

  return (
    <div>
      {/* Description */}
      {tool.description && (
        <div style={{ marginBottom: 16 }}>
          <Text
            style={{
              lineHeight: 1.6,
              whiteSpace: "pre-wrap",
            }}
          >
            {tool.description}
          </Text>
        </div>
      )}

      {/* Parameters Table */}
      {parameterRows.length > 0 && (
        <div>
          <Text
            type="secondary"
            style={{
              fontSize: 12,
              display: "block",
              marginBottom: 8,
            }}
          >
            {t("viewLogs.formattedToolView.parametersLabel")}
          </Text>
          <Table dataSource={parameterRows} columns={columns} pagination={false} size="small" bordered />
        </div>
      )}

      {/* If tool was called, show the arguments used */}
      {tool.called && tool.callData && (
        <div style={{ marginTop: 16 }}>
          <Text
            type="secondary"
            style={{
              fontSize: 12,
              display: "block",
              marginBottom: 8,
            }}
          >
            {t("viewLogs.formattedToolView.calledWithLabel")}
          </Text>
          <div
            style={{
              background: "#f6ffed",
              border: "1px solid #b7eb8f",
              borderRadius: 4,
              padding: 12,
            }}
          >
            <pre
              style={{
                margin: 0,
                fontSize: 12,
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
              }}
            >
              {JSON.stringify(tool.callData.arguments, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}
