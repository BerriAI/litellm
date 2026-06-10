import { CodeOutlined, PlayCircleOutlined } from "@ant-design/icons";
import { Alert, Button, Card, Input, Space, Tabs, Typography } from "antd";
import { useTranslation } from "react-i18next";
import ModelSelector from "@/components/common_components/ModelSelector";
import { TestResult } from "./semanticFilterTestUtils";

interface MCPSemanticFilterTestPanelProps {
  accessToken: string | null;
  testQuery: string;
  setTestQuery: (value: string) => void;
  testModel: string;
  setTestModel: (value: string) => void;
  isTesting: boolean;
  onTest: () => void;
  filterEnabled: boolean;
  testResult: TestResult | null;
  curlCommand: string;
}

export default function MCPSemanticFilterTestPanel({
  accessToken,
  testQuery,
  setTestQuery,
  testModel,
  setTestModel,
  isTesting,
  onTest,
  filterEnabled,
  testResult,
  curlCommand,
}: MCPSemanticFilterTestPanelProps) {
  const { t } = useTranslation();
  return (
    <Card title={t("settingsPages.mcpSemanticFilterTestPanel.testConfigCardTitle")} style={{ marginBottom: 16 }}>
      <Tabs
        defaultActiveKey="test"
        items={[
          {
            key: "test",
            label: t("settingsPages.mcpSemanticFilterTestPanel.testTabLabel"),
            children: (
              <Space direction="vertical" style={{ width: "100%" }} size="large">
                <div>
                  <Typography.Text strong style={{ display: "block", marginBottom: 8 }}>
                    <PlayCircleOutlined /> {t("settingsPages.mcpSemanticFilterTestPanel.testQueryLabel")}
                  </Typography.Text>
                  <Input.TextArea
                    placeholder={t("settingsPages.mcpSemanticFilterTestPanel.testQueryPlaceholder")}
                    value={testQuery}
                    onChange={(e) => setTestQuery(e.target.value)}
                    rows={4}
                    disabled={isTesting}
                  />
                </div>

                <div>
                  <ModelSelector
                    accessToken={accessToken || ""}
                    value={testModel}
                    onChange={setTestModel}
                    disabled={isTesting}
                    showLabel={true}
                    labelText={t("settingsPages.mcpSemanticFilterTestPanel.selectModelLabel")}
                  />
                </div>

                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={onTest}
                  loading={isTesting}
                  disabled={!testQuery || !testModel || !filterEnabled}
                  block
                >
                  {t("settingsPages.mcpSemanticFilterTestPanel.testFilterButton")}
                </Button>

                {!filterEnabled && (
                  <Alert
                    type="warning"
                    message={t("settingsPages.mcpSemanticFilterTestPanel.filterDisabledTitle")}
                    description={t("settingsPages.mcpSemanticFilterTestPanel.filterDisabledDesc")}
                    showIcon
                  />
                )}

                {testResult && (
                  <div>
                    <Typography.Title level={5}>
                      {t("settingsPages.mcpSemanticFilterTestPanel.resultsTitle")}
                    </Typography.Title>
                    <Alert
                      type="success"
                      message={t("settingsPages.mcpSemanticFilterTestPanel.toolsSelectedMessage", {
                        selectedTools: testResult.selectedTools,
                      })}
                      description={t("settingsPages.mcpSemanticFilterTestPanel.filteredFromMessage", {
                        totalTools: testResult.totalTools,
                      })}
                      showIcon
                      style={{ marginBottom: 16 }}
                    />
                    <div>
                      <Typography.Text strong style={{ display: "block", marginBottom: 8 }}>
                        {t("settingsPages.mcpSemanticFilterTestPanel.selectedToolsLabel")}
                      </Typography.Text>
                      <ul style={{ paddingLeft: 20, margin: 0 }}>
                        {testResult.tools.map((tool, index) => (
                          <li key={index} style={{ marginBottom: 4 }}>
                            <Typography.Text>{tool}</Typography.Text>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}
              </Space>
            ),
          },
          {
            key: "api",
            label: t("settingsPages.mcpSemanticFilterTestPanel.apiTabLabel"),
            children: (
              <div>
                <Space style={{ marginBottom: 8 }}>
                  <CodeOutlined />
                  <Typography.Text strong>
                    {t("settingsPages.mcpSemanticFilterTestPanel.apiUsageTitle")}
                  </Typography.Text>
                </Space>
                <Typography.Text type="secondary" style={{ display: "block", marginBottom: 8 }}>
                  {t("settingsPages.mcpSemanticFilterTestPanel.apiUsageDesc")}
                </Typography.Text>
                <Typography.Text strong style={{ display: "block", marginBottom: 8 }}>
                  {t("settingsPages.mcpSemanticFilterTestPanel.responseHeadersLabel")}
                </Typography.Text>
                <ul style={{ paddingLeft: 20, margin: "0 0 12px 0" }}>
                  <li>
                    <Typography.Text>
                      {t("settingsPages.mcpSemanticFilterTestPanel.semanticFilterHeader")}
                    </Typography.Text>
                    <Typography.Text type="secondary" style={{ display: "block" }}>
                      {t("settingsPages.mcpSemanticFilterTestPanel.semanticFilterHeaderExample")}
                    </Typography.Text>
                  </li>
                  <li>
                    <Typography.Text>
                      {t("settingsPages.mcpSemanticFilterTestPanel.semanticFilterToolsHeader")}
                    </Typography.Text>
                    <Typography.Text type="secondary" style={{ display: "block" }}>
                      {t("settingsPages.mcpSemanticFilterTestPanel.semanticFilterToolsHeaderExample")}
                    </Typography.Text>
                  </li>
                </ul>
                <pre
                  style={{
                    background: "#f5f5f5",
                    padding: 12,
                    borderRadius: 4,
                    overflow: "auto",
                    fontSize: 12,
                    margin: 0,
                  }}
                >
                  {curlCommand}
                </pre>
              </div>
            ),
          },
        ]}
      />
    </Card>
  );
}
