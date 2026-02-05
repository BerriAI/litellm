import { CodeOutlined, PlayCircleOutlined } from "@ant-design/icons";
import { Alert, Button, Card, Input, Space, Typography } from "antd";
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
  showCurl: boolean;
  setShowCurl: (value: boolean) => void;
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
  showCurl,
  setShowCurl,
  curlCommand,
}: MCPSemanticFilterTestPanelProps) {
  return (
    <>
      <Card title="Test Configuration" style={{ marginBottom: 16 }}>
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
      </Card>

      <Card
        title={
          <Space>
            <CodeOutlined />
            <Typography.Text>API Usage</Typography.Text>
          </Space>
        }
      >
        <Typography.Text type="secondary" style={{ display: "block", marginBottom: 12 }}>
          Use this curl command to test the semantic filter with your current configuration.
        </Typography.Text>
        <Button
          type="link"
          onClick={() => setShowCurl(!showCurl)}
          style={{ padding: 0, marginBottom: 12 }}
        >
          {showCurl ? "Hide" : "Show"} curl command
        </Button>
        {showCurl && (
          <pre
            style={{
              background: "#f5f5f5",
              padding: 12,
              borderRadius: 4,
              overflow: "auto",
              fontSize: 12,
            }}
          >
            {curlCommand}
          </pre>
        )}
      </Card>
    </>
  );
}
