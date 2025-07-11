import React from "react";
import { Button, Callout, TextInput } from "@tremor/react";
import { MCPTool, InputSchema } from "./types";
import { Modal, Form, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";

const AuthBanner = ({ needsAuth, authValue }: { needsAuth: boolean; authValue?: string | null }) => {
  if (!needsAuth || (needsAuth && authValue)) {
    return (
      <Callout title="Authentication" color="green" className="mb-4">
        This tool does not require authentication or has authentication added.
      </Callout>
    );
  }

  if (needsAuth && !authValue) {
    return (
      <Callout title="Authentication required" color="yellow" className="mb-4">
        Please provide authentication details if this tool call requires auth.
      </Callout>
    );
  }
  return null;
};

export function ToolTestPanel({
  tool,
  needsAuth,
  authValue,
  onSubmit,
  isLoading,
  result,
  error,
  onClose,
}: {
  tool: MCPTool;
  needsAuth: boolean;
  authValue?: string | null;
  onSubmit: (args: Record<string, any>) => void;
  isLoading: boolean;
  result: any | null;
  error: Error | null;
  onClose: () => void;
}) {
  const [form] = Form.useForm();

  // Create a placeholder schema if we only have the "tool_input_schema" string
  const schema: InputSchema = React.useMemo(() => {
    if (typeof tool.inputSchema === "string") {
      // Default schema with a single text field
      return {
        type: "object",
        properties: {
          input: {
            type: "string",
            description: "Input for this tool",
          },
        },
        required: ["input"],
      };
    }
    return tool.inputSchema as InputSchema;
  }, [tool.inputSchema]);

  const handleSubmit = (values: Record<string, any>) => {
    onSubmit(values);
  };

  return (
    <Modal
      title={
        <div className="flex items-center space-x-3 pb-4 border-b border-gray-100">
          {tool.mcp_info.logo_url && (
            <img
              src={tool.mcp_info.logo_url}
              alt={`${tool.mcp_info.server_name} logo`}
              className="w-8 h-8 object-contain"
              style={{ 
                height: '20px', 
                width: '20px', 
                marginRight: '8px',
                objectFit: 'contain'
              }}
            />
          )}
          <div className="flex-1 min-w-0">
            <h2 className="text-xl font-semibold text-gray-900">
              Test Tool: <span className="font-mono text-blue-600">{tool.name}</span>
            </h2>
            <p className="text-sm text-gray-600 mt-1">{tool.description}</p>
            <p className="text-xs text-gray-500 mt-1">Provider: {tool.mcp_info.server_name}</p>
          </div>
        </div>
      }
      open={true}
      width={1200}
      onCancel={onClose}
      footer={null}
      className="top-8"
      styles={{
        body: { padding: '24px' },
        header: { padding: '24px 24px 0 24px', border: 'none' },
      }}
    >
      <div className="mt-6">
        {/* Auth Banner */}
        <AuthBanner needsAuth={needsAuth} authValue={authValue} />
        
        {/* Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Form Section */}
          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-6 flex items-center">
              Input Parameters
              <Tooltip title="Configure the input parameters for this tool call">
                <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
              </Tooltip>
            </h3>
            
            <Form form={form} onFinish={handleSubmit} layout="vertical" className="space-y-6">
              <div className="space-y-4">
                {typeof tool.inputSchema === "string" ? (
                  <div>
                    <p className="text-xs text-gray-500 mb-3">This tool uses a dynamic input schema.</p>
                    <Form.Item
                      label={
                        <span className="text-sm font-medium text-gray-700">
                          Input <span className="text-red-500">*</span>
                        </span>
                      }
                      name="input"
                      rules={[{ required: true, message: "Please enter input for this tool" }]}
                    >
                      <TextInput
                        placeholder="Enter input for this tool"
                        className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                      />
                    </Form.Item>
                  </div>
                ) : schema.properties === undefined ? (
                  <div className="text-center py-8 text-gray-500">
                    <p className="text-sm">This tool requires no input parameters.</p>
                  </div>
                ) : (
                  Object.entries(schema.properties).map(([key, prop]) => (
                    <Form.Item
                      key={key}
                      label={
                        <span className="text-sm font-medium text-gray-700 flex items-center">
                          {key}{" "}
                          {schema.required?.includes(key) && <span className="text-red-500">*</span>}
                          {prop.description && (
                            <Tooltip title={prop.description}>
                              <InfoCircleOutlined className="ml-2 text-gray-400 hover:text-gray-600" />
                            </Tooltip>
                          )}
                        </span>
                      }
                      name={key}
                      rules={[
                        {
                          required: schema.required?.includes(key),
                          message: `Please enter ${key}`,
                        },
                      ]}
                    >
                      {prop.type === "string" && (
                        <TextInput
                          placeholder={prop.description || `Enter ${key}`}
                          className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                        />
                      )}

                      {prop.type === "number" && (
                        <input
                          type="number"
                          placeholder={prop.description || `Enter ${key}`}
                          className="w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
                        />
                      )}

                      {prop.type === "boolean" && (
                        <div className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                          />
                          <span className="text-sm text-gray-600">Enable</span>
                        </div>
                      )}
                    </Form.Item>
                  ))
                )}
              </div>

              <div className="pt-4 border-t border-gray-200">
                <Button
                  onClick={() => form.submit()}
                  disabled={isLoading}
                  variant="primary"
                  className="w-full"
                  loading={isLoading}
                >
                  {isLoading ? "Calling..." : "Call Tool"}
                </Button>
              </div>
            </Form>
          </div>

          {/* Result Section */}
          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-6">Result</h3>

            <div className="min-h-[400px] max-h-[600px] overflow-y-auto">
              {isLoading && (
                <div className="flex flex-col justify-center items-center h-full text-gray-500">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-3"></div>
                  <p className="text-sm">Calling tool...</p>
                </div>
              )}

              {error && (
                <div className="bg-red-50 border border-red-200 text-red-800 p-4 rounded-md">
                  <p className="font-medium mb-2">Error</p>
                  <div className="bg-white border border-red-200 rounded p-3 max-h-64 overflow-y-auto">
                    <pre className="text-xs whitespace-pre-wrap text-red-700">{error.message}</pre>
                  </div>
                </div>
              )}

              {result && !isLoading && !error && (
                <div className="space-y-4">
                  {result.map((content: any, idx: number) => (
                    <div key={idx}>
                      {content.type === "text" && (
                        <div className="bg-white border border-gray-200 p-4 rounded-md">
                          <div className="text-xs text-gray-500 mb-2 font-medium">Text Response</div>
                          <div className="prose prose-sm max-w-none">
                            <pre className="whitespace-pre-wrap text-sm text-gray-900 font-sans">
                              {content.text}
                            </pre>
                          </div>
                        </div>
                      )}

                      {content.type === "image" && content.url && (
                        <div className="bg-white border border-gray-200 p-4 rounded-md">
                          <div className="text-xs text-gray-500 mb-2 font-medium">Image Response</div>
                          <img
                            src={content.url}
                            alt="Tool result"
                            className="max-w-full h-auto rounded border"
                          />
                        </div>
                      )}

                      {content.type === "embedded_resource" && (
                        <div className="bg-white border border-gray-200 p-4 rounded-md">
                          <div className="text-xs text-gray-500 mb-2 font-medium">Embedded Resource</div>
                          <p className="text-sm font-medium text-gray-900">Resource Type: {content.resource_type}</p>
                          {content.url && (
                            <a
                              href={content.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center text-sm text-blue-600 hover:text-blue-800 hover:underline mt-2"
                            >
                              View Resource
                              <svg className="ml-1 h-3 w-3" fill="currentColor" viewBox="0 0 20 20">
                                <path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z" />
                                <path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z" />
                              </svg>
                            </a>
                          )}
                        </div>
                      )}
                    </div>
                  ))}

                  <div className="border-t border-gray-200 pt-4">
                    <details className="text-xs">
                      <summary className="cursor-pointer text-gray-500 hover:text-gray-700 font-medium">
                        View Raw JSON Response
                      </summary>
                      <div className="mt-3 bg-gray-900 text-green-400 p-3 rounded-md overflow-x-auto">
                        <pre className="text-xs">{JSON.stringify(result, null, 2)}</pre>
                      </div>
                    </details>
                  </div>
                </div>
              )}

              {!result && !isLoading && !error && (
                <div className="flex flex-col justify-center items-center h-full text-gray-500">
                  <div className="text-center">
                    <svg className="mx-auto h-12 w-12 text-gray-300 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    <p className="text-sm">The result will appear here after you call the tool.</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </Modal>
  );
} 