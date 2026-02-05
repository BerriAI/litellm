import { CodeOutlined, PlayCircleOutlined } from "@ant-design/icons";
import { Alert, Button, Card, Input, Space, Tabs, Typography } from "antd";
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
  return (
    <Card title="Test Configuration" style={{ marginBottom: 16 }}>
      <Tabs
        defaultActiveKey="test"
        items={[
          {
            key: "test",
            label: "Test",
            children: (
              <Space direction="vertical" style={{ width: "100%" }} size="large">
          <div>
            <Typography.Text strong style={{ display: "block", marginBottom: 8 }}>
              <PlayCircleOutlined /> Test Query
            </Typography.Text>
            <Input.TextArea
              placeholder="Enter a test query to see which tools would be selected..."
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
              labelText="Select Model"
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
            Test Filter
          </Button>

          {!filterEnabled && (
            <Alert
              type="warning"
              message="Semantic filtering is disabled"
              description="Enable semantic filtering and save settings to test the filter."
              showIcon
            />
          )}

          {testResult && (
            <div>
              <Typography.Title level={5}>Results</Typography.Title>
              <Alert
                type="success"
                message={`${testResult.selectedTools} tools selected`}
                description={`Filtered from ${testResult.totalTools} available tools`}
                showIcon
                style={{ marginBottom: 16 }}
              />
              <div>
                <Typography.Text strong style={{ display: "block", marginBottom: 8 }}>
                  Selected Tools:
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
            label: "API Usage",
            children: (
              <div>
                <Space style={{ marginBottom: 8 }}>
                  <CodeOutlined />
                  <Typography.Text strong>API Usage</Typography.Text>
                </Space>
                <Typography.Text type="secondary" style={{ display: "block", marginBottom: 8 }}>
                  Use this curl command to test the semantic filter with your current configuration.
                </Typography.Text>
            <Typography.Text strong style={{ display: "block", marginBottom: 8 }}>
              Response headers to check:
            </Typography.Text>
            <ul style={{ paddingLeft: 20, margin: "0 0 12px 0" }}>
              <li>
                <Typography.Text>
                  x-litellm-semantic-filter: shows total tools → selected tools
                </Typography.Text>
                <Typography.Text type="secondary" style={{ display: "block" }}>
                  Example: 10→3
                </Typography.Text>
              </li>
              <li>
                <Typography.Text>
                  x-litellm-semantic-filter-tools: CSV of selected tool names
                </Typography.Text>
                <Typography.Text type="secondary" style={{ display: "block" }}>
                  Example: wikipedia-fetch,github-search,slack-post
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
