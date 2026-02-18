import React, { useState, useCallback } from "react";
import { Card, Text, Button } from "@tremor/react";
import { Collapse, Typography, Tooltip, Spin, Alert, Tabs } from "antd";
import {
  PlayCircleOutlined,
  InfoCircleOutlined,
  CodeOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  EditOutlined,
  BookOutlined,
  ExperimentOutlined,
} from "@ant-design/icons";
import NotificationsManager from "../../molecules/notifications_manager";
import { testCustomCodeGuardrail } from "../../networking";
import CustomCodeEditor from "./CustomCodeEditor";
import { CUSTOM_CODE_PRIMITIVES, CUSTOM_CODE_EXAMPLES, DEFAULT_CUSTOM_CODE } from "./custom_code_constants";

const { Panel } = Collapse;
const { Paragraph } = Typography;

interface CustomCodePlaygroundProps {
  accessToken: string | null;
  initialCode?: string;
  onCodeChange?: (code: string) => void;
  showTestingPanel?: boolean;
}

interface TestResult {
  action: "allow" | "block" | "modify";
  reason?: string;
  modified_texts?: string[];
  modified_images?: string[];
  modified_tool_calls?: any[];
  execution_time_ms?: number;
  error?: string;
}

const CustomCodePlayground: React.FC<CustomCodePlaygroundProps> = ({
  accessToken,
  initialCode = DEFAULT_CUSTOM_CODE,
  onCodeChange,
  showTestingPanel = true,
}) => {
  const [customCode, setCustomCode] = useState(initialCode);
  const [testInput, setTestInput] = useState(
    JSON.stringify(
      {
        texts: ["Hello, my SSN is 123-45-6789"],
        images: [],
        tools: [],
        tool_calls: [],
        structured_messages: [
          { role: "user", content: "Hello, my SSN is 123-45-6789" },
        ],
        model: "gpt-4",
      },
      null,
      2
    )
  );
  const [requestData, setRequestData] = useState(
    JSON.stringify(
      {
        model: "gpt-4",
        user_id: "test-user",
        team_id: "test-team",
        end_user_id: "end-user-123",
        metadata: {},
      },
      null,
      2
    )
  );
  const [inputType, setInputType] = useState<"request" | "response">("request");
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [isTesting, setIsTesting] = useState(false);
  const [activeTab, setActiveTab] = useState<string>("editor");

  const handleCodeChange = useCallback(
    (code: string) => {
      setCustomCode(code);
      onCodeChange?.(code);
    },
    [onCodeChange]
  );

  const handleRunTest = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
      return;
    }

    setIsTesting(true);
    setTestResult(null);

    try {
      let parsedInputs: any;
      let parsedRequestData: any;

      try {
        parsedInputs = JSON.parse(testInput);
      } catch (e) {
        throw new Error("Invalid JSON in test input");
      }

      try {
        parsedRequestData = JSON.parse(requestData);
      } catch (e) {
        throw new Error("Invalid JSON in request data");
      }

      const response = await testCustomCodeGuardrail(accessToken, {
        custom_code: customCode,
        test_input: parsedInputs,
        input_type: inputType,
        request_data: parsedRequestData,
      });

      if (response.success && response.result) {
        setTestResult(response.result);

        if (response.result.action === "allow") {
          NotificationsManager.success("Guardrail allowed the request");
        } else if (response.result.action === "block") {
          NotificationsManager.fromBackend(`Guardrail blocked: ${response.result.reason || "No reason provided"}`);
        } else if (response.result.action === "modify") {
          NotificationsManager.success("Guardrail modified the content");
        }
      } else if (response.error) {
        setTestResult({
          action: "block",
          error: response.error,
        });
        NotificationsManager.fromBackend(`Test failed: ${response.error}`);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unknown error";
      setTestResult({
        action: "block",
        error: errorMessage,
      });
      NotificationsManager.fromBackend(`Test failed: ${errorMessage}`);
    } finally {
      setIsTesting(false);
    }
  };

  const loadExample = (exampleKey: keyof typeof CUSTOM_CODE_EXAMPLES) => {
    setCustomCode(CUSTOM_CODE_EXAMPLES[exampleKey]);
    onCodeChange?.(CUSTOM_CODE_EXAMPLES[exampleKey]);
    setActiveTab("editor");
  };

  const renderTestResult = () => {
    if (!testResult) return null;

    const isError = !!testResult.error;
    const isAllow = testResult.action === "allow";
    const isBlock = testResult.action === "block";
    const isModify = testResult.action === "modify";

    return (
      <div className="mt-4">
        <Text className="font-medium text-gray-700 block mb-2">Test Result</Text>
        <div
          className={`rounded-lg p-4 border ${
            isError
              ? "bg-red-50 border-red-200"
              : isAllow
              ? "bg-green-50 border-green-200"
              : isBlock
              ? "bg-orange-50 border-orange-200"
              : "bg-blue-50 border-blue-200"
          }`}
        >
          <div className="flex items-center gap-2 mb-2">
            {isError ? (
              <CloseCircleOutlined className="text-red-500 text-lg" />
            ) : isAllow ? (
              <CheckCircleOutlined className="text-green-500 text-lg" />
            ) : isBlock ? (
              <CloseCircleOutlined className="text-orange-500 text-lg" />
            ) : (
              <EditOutlined className="text-blue-500 text-lg" />
            )}
            <span
              className={`font-semibold ${
                isError
                  ? "text-red-700"
                  : isAllow
                  ? "text-green-700"
                  : isBlock
                  ? "text-orange-700"
                  : "text-blue-700"
              }`}
            >
              {isError ? "Error" : testResult.action.toUpperCase()}
            </span>
            {testResult.execution_time_ms && (
              <span className="text-xs text-gray-500 ml-auto">
                {testResult.execution_time_ms.toFixed(2)}ms
              </span>
            )}
          </div>

          {isError && (
            <Alert type="error" message={testResult.error} className="mt-2" />
          )}

          {isBlock && testResult.reason && (
            <Paragraph className="text-orange-700 mb-0 mt-2">
              <strong>Reason:</strong> {testResult.reason}
            </Paragraph>
          )}

          {isModify && testResult.modified_texts && testResult.modified_texts.length > 0 && (
            <div className="mt-2">
              <Text className="font-medium text-blue-700 block mb-1">Modified Texts:</Text>
              <pre className="bg-white rounded p-2 text-xs overflow-auto max-h-32 border border-blue-100">
                {JSON.stringify(testResult.modified_texts, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderPrimitivesReference = () => (
    <div className="space-y-4">
      {Object.entries(CUSTOM_CODE_PRIMITIVES).map(([category, primitives]) => (
        <div key={category}>
          <Text className="font-semibold text-gray-700 block mb-2">{category}</Text>
          <div className="bg-gray-50 rounded-lg p-3">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-1 pr-4 font-medium text-gray-600">Function</th>
                  <th className="text-left py-1 font-medium text-gray-600">Description</th>
                </tr>
              </thead>
              <tbody>
                {primitives.map((primitive) => (
                  <tr key={primitive.name} className="border-b border-gray-100 last:border-0">
                    <td className="py-1.5 pr-4">
                      <code className="text-xs bg-blue-50 text-blue-700 px-1.5 py-0.5 rounded font-mono">
                        {primitive.signature}
                      </code>
                    </td>
                    <td className="py-1.5 text-gray-600">{primitive.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}

      <div>
        <Text className="font-semibold text-gray-700 block mb-2">Return Values</Text>
        <div className="bg-gray-50 rounded-lg p-3">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-1 pr-4 font-medium text-gray-600">Function</th>
                <th className="text-left py-1 font-medium text-gray-600">Description</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-100">
                <td className="py-1.5 pr-4"><code className="text-xs bg-green-100 text-green-700 px-1.5 py-0.5 rounded font-mono">allow()</code></td>
                <td className="py-1.5 text-gray-600">Let request/response through</td>
              </tr>
              <tr className="border-b border-gray-100">
                <td className="py-1.5 pr-4"><code className="text-xs bg-red-100 text-red-700 px-1.5 py-0.5 rounded font-mono">block(reason)</code></td>
                <td className="py-1.5 text-gray-600">Reject with message</td>
              </tr>
              <tr className="border-b border-gray-100">
                <td className="py-1.5 pr-4"><code className="text-xs bg-yellow-100 text-yellow-700 px-1.5 py-0.5 rounded font-mono">modify(texts=[], images=[], tool_calls=[])</code></td>
                <td className="py-1.5 text-gray-600">Transform content</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  const renderInputParamsReference = () => (
    <div className="space-y-4">
      <div>
        <Text className="font-semibold text-gray-700 block mb-2">`inputs` Parameter</Text>
        <div className="bg-gray-50 rounded-lg p-3">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-1 pr-4 font-medium text-gray-600">Field</th>
                <th className="text-left py-1 pr-4 font-medium text-gray-600">Type</th>
                <th className="text-left py-1 font-medium text-gray-600">Description</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-100">
                <td className="py-1.5 pr-4"><code className="text-xs bg-gray-200 px-1 rounded font-mono">texts</code></td>
                <td className="py-1.5 pr-4 text-gray-500">List[str]</td>
                <td className="py-1.5 text-gray-600">Extracted text from the request/response</td>
              </tr>
              <tr className="border-b border-gray-100">
                <td className="py-1.5 pr-4"><code className="text-xs bg-gray-200 px-1 rounded font-mono">images</code></td>
                <td className="py-1.5 pr-4 text-gray-500">List[str]</td>
                <td className="py-1.5 text-gray-600">Extracted images (for image guardrails)</td>
              </tr>
              <tr className="border-b border-gray-100">
                <td className="py-1.5 pr-4"><code className="text-xs bg-gray-200 px-1 rounded font-mono">tools</code></td>
                <td className="py-1.5 pr-4 text-gray-500">List[dict]</td>
                <td className="py-1.5 text-gray-600">Tools sent to the LLM</td>
              </tr>
              <tr className="border-b border-gray-100">
                <td className="py-1.5 pr-4"><code className="text-xs bg-gray-200 px-1 rounded font-mono">tool_calls</code></td>
                <td className="py-1.5 pr-4 text-gray-500">List[dict]</td>
                <td className="py-1.5 text-gray-600">Tool calls returned from the LLM</td>
              </tr>
              <tr className="border-b border-gray-100">
                <td className="py-1.5 pr-4"><code className="text-xs bg-gray-200 px-1 rounded font-mono">structured_messages</code></td>
                <td className="py-1.5 pr-4 text-gray-500">List[dict]</td>
                <td className="py-1.5 text-gray-600">Full messages with role info</td>
              </tr>
              <tr className="border-b border-gray-100">
                <td className="py-1.5 pr-4"><code className="text-xs bg-gray-200 px-1 rounded font-mono">model</code></td>
                <td className="py-1.5 pr-4 text-gray-500">str</td>
                <td className="py-1.5 text-gray-600">The model being used</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <Text className="font-semibold text-gray-700 block mb-2">`request_data` Parameter</Text>
        <div className="bg-gray-50 rounded-lg p-3">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-1 pr-4 font-medium text-gray-600">Field</th>
                <th className="text-left py-1 pr-4 font-medium text-gray-600">Type</th>
                <th className="text-left py-1 font-medium text-gray-600">Description</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-100">
                <td className="py-1.5 pr-4"><code className="text-xs bg-gray-200 px-1 rounded font-mono">model</code></td>
                <td className="py-1.5 pr-4 text-gray-500">str</td>
                <td className="py-1.5 text-gray-600">Model name</td>
              </tr>
              <tr className="border-b border-gray-100">
                <td className="py-1.5 pr-4"><code className="text-xs bg-gray-200 px-1 rounded font-mono">user_id</code></td>
                <td className="py-1.5 pr-4 text-gray-500">str</td>
                <td className="py-1.5 text-gray-600">User ID from API key</td>
              </tr>
              <tr className="border-b border-gray-100">
                <td className="py-1.5 pr-4"><code className="text-xs bg-gray-200 px-1 rounded font-mono">team_id</code></td>
                <td className="py-1.5 pr-4 text-gray-500">str</td>
                <td className="py-1.5 text-gray-600">Team ID from API key</td>
              </tr>
              <tr className="border-b border-gray-100">
                <td className="py-1.5 pr-4"><code className="text-xs bg-gray-200 px-1 rounded font-mono">end_user_id</code></td>
                <td className="py-1.5 pr-4 text-gray-500">str</td>
                <td className="py-1.5 text-gray-600">End user ID</td>
              </tr>
              <tr className="border-b border-gray-100">
                <td className="py-1.5 pr-4"><code className="text-xs bg-gray-200 px-1 rounded font-mono">metadata</code></td>
                <td className="py-1.5 pr-4 text-gray-500">dict</td>
                <td className="py-1.5 text-gray-600">Request metadata</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  const renderExamplesTab = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      <button
        onClick={() => loadExample("blockSSN")}
        className="text-left p-4 border border-gray-200 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors"
      >
        <Text className="font-medium text-gray-800 block mb-1">🔒 Block PII (SSN)</Text>
        <Text className="text-xs text-gray-500">Detect and block Social Security Numbers</Text>
      </button>
      <button
        onClick={() => loadExample("redactEmail")}
        className="text-left p-4 border border-gray-200 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors"
      >
        <Text className="font-medium text-gray-800 block mb-1">📧 Redact Emails</Text>
        <Text className="text-xs text-gray-500">Replace email addresses with [EMAIL REDACTED]</Text>
      </button>
      <button
        onClick={() => loadExample("blockSQL")}
        className="text-left p-4 border border-gray-200 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors"
      >
        <Text className="font-medium text-gray-800 block mb-1">🛡️ Block SQL Injection</Text>
        <Text className="text-xs text-gray-500">Prevent SQL code in requests</Text>
      </button>
      <button
        onClick={() => loadExample("validateJSON")}
        className="text-left p-4 border border-gray-200 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors"
      >
        <Text className="font-medium text-gray-800 block mb-1">✅ Validate JSON Response</Text>
        <Text className="text-xs text-gray-500">Ensure responses have required fields</Text>
      </button>
      <button
        onClick={() => loadExample("checkURLs")}
        className="text-left p-4 border border-gray-200 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors"
      >
        <Text className="font-medium text-gray-800 block mb-1">🔗 Check URLs</Text>
        <Text className="text-xs text-gray-500">Validate all URLs in responses</Text>
      </button>
      <button
        onClick={() => loadExample("combined")}
        className="text-left p-4 border border-gray-200 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors"
      >
        <Text className="font-medium text-gray-800 block mb-1">🔄 Combined Checks</Text>
        <Text className="text-xs text-gray-500">Multiple checks with redaction and blocking</Text>
      </button>
    </div>
  );

  const tabItems = [
    {
      key: "editor",
      label: (
        <span className="flex items-center gap-1.5">
          <CodeOutlined />
          Code Editor
        </span>
      ),
      children: (
        <div className="space-y-4">
          <CustomCodeEditor value={customCode} onChange={handleCodeChange} height="350px" />
          
          <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-sm text-yellow-800">
            <strong>⚠️ Sandbox Restrictions:</strong> No imports, no file I/O, no network access, no exec() or eval().
            Only LiteLLM-provided primitives are available.
          </div>
        </div>
      ),
    },
    {
      key: "examples",
      label: (
        <span className="flex items-center gap-1.5">
          <BookOutlined />
          Examples
        </span>
      ),
      children: renderExamplesTab(),
    },
    {
      key: "primitives",
      label: (
        <span className="flex items-center gap-1.5">
          <InfoCircleOutlined />
          Primitives Reference
        </span>
      ),
      children: renderPrimitivesReference(),
    },
    {
      key: "params",
      label: (
        <span className="flex items-center gap-1.5">
          <InfoCircleOutlined />
          Input Parameters
        </span>
      ),
      children: renderInputParamsReference(),
    },
  ];

  if (showTestingPanel) {
    tabItems.push({
      key: "test",
      label: (
        <span className="flex items-center gap-1.5">
          <ExperimentOutlined />
          Test
        </span>
      ),
      children: (
        <div className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <Text className="font-medium text-gray-700">Test Input (inputs parameter)</Text>
                <Tooltip title="This represents the 'inputs' parameter passed to your apply_guardrail function">
                  <InfoCircleOutlined className="text-gray-400" />
                </Tooltip>
              </div>
              <textarea
                value={testInput}
                onChange={(e) => setTestInput(e.target.value)}
                className="w-full h-48 p-3 font-mono text-sm border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Enter test input JSON..."
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-2">
                <Text className="font-medium text-gray-700">Request Data (request_data parameter)</Text>
                <Tooltip title="This represents the 'request_data' parameter passed to your apply_guardrail function">
                  <InfoCircleOutlined className="text-gray-400" />
                </Tooltip>
              </div>
              <textarea
                value={requestData}
                onChange={(e) => setRequestData(e.target.value)}
                className="w-full h-48 p-3 font-mono text-sm border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="Enter request data JSON..."
              />
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Text className="text-sm text-gray-600">Input Type:</Text>
              <select
                value={inputType}
                onChange={(e) => setInputType(e.target.value as "request" | "response")}
                className="border border-gray-200 rounded px-3 py-1.5 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="request">request</option>
                <option value="response">response</option>
              </select>
            </div>

            <Button
              onClick={handleRunTest}
              disabled={!accessToken || isTesting}
              icon={isTesting ? undefined : PlayCircleOutlined}
              className="ml-auto"
            >
              {isTesting ? (
                <span className="flex items-center gap-2">
                  <Spin size="small" /> Running Test...
                </span>
              ) : (
                "Run Test"
              )}
            </Button>
          </div>

          {renderTestResult()}
        </div>
      ),
    });
  }

  return (
    <Card className="p-0 overflow-hidden">
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={tabItems}
        className="custom-code-playground-tabs"
        tabBarStyle={{ padding: "0 16px", marginBottom: 0 }}
      />
      <div className="p-4">
        {tabItems.find(tab => tab.key === activeTab)?.children}
      </div>
      <style>{`
        .custom-code-playground-tabs .ant-tabs-nav {
          background: #f9fafb;
          border-bottom: 1px solid #e5e7eb;
        }
        .custom-code-playground-tabs .ant-tabs-tab {
          padding: 12px 16px;
        }
        .custom-code-playground-tabs .ant-tabs-tab-active {
          background: white;
        }
      `}</style>
    </Card>
  );
};

export default CustomCodePlayground;
