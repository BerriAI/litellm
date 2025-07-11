import React from "react";
import { Button, Callout, TextInput } from "@tremor/react";
import { MCPTool, InputSchema } from "./types";
import { Modal, Form, Tooltip, message } from "antd";
import { InfoCircleOutlined, ClockCircleOutlined } from "@ant-design/icons";

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
  const [viewMode, setViewMode] = React.useState<'formatted' | 'json'>('formatted');
  const [startTime, setStartTime] = React.useState<number | null>(null);
  const [duration, setDuration] = React.useState<number | null>(null);

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
    const start = Date.now();
    setStartTime(start);
    setDuration(null);
    onSubmit(values);
  };

  // Track when result changes to calculate duration
  React.useEffect(() => {
    if (startTime && (result || error)) {
      const endTime = Date.now();
      setDuration(endTime - startTime);
    }
  }, [result, error, startTime]);

  const copyToClipboard = async (text: string) => {
    try {
      // Try modern clipboard API first
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return true;
      } else {
        // Fallback for non-secure contexts (like 0.0.0.0)
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.opacity = '0';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        const successful = document.execCommand('copy');
        document.body.removeChild(textArea);
        
        if (!successful) {
          throw new Error('execCommand failed');
        }
        return true;
      }
    } catch (error) {
      console.error('Copy failed:', error);
      return false;
    }
  };

  const handleCopyResult = async () => {
    const success = await copyToClipboard(JSON.stringify(result, null, 2));
    if (success) {
      message.success('Result copied to clipboard');
    } else {
      message.error('Failed to copy result');
    }
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
        <div className={`grid gap-8 ${result || error ? 'grid-cols-1' : 'grid-cols-1 lg:grid-cols-2'}`}>
          {/* Form Section */}
          <div className="space-y-6">
            <div className="bg-white border border-gray-200 rounded-lg p-6 shadow-sm">
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-lg font-semibold text-gray-900">Input Parameters</h3>
                <Tooltip title="Configure the input parameters for this tool call">
                  <InfoCircleOutlined className="text-gray-400 hover:text-gray-600" />
                </Tooltip>
              </div>
              
              <Form form={form} onFinish={handleSubmit} layout="vertical" className={`${result || error ? 'space-y-2' : 'space-y-4'}`}>
                {typeof tool.inputSchema === "string" ? (
                  <div className={`${result || error ? 'space-y-2' : 'space-y-4'}`}>
                    {!(result || error) && (
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <div className="flex items-start space-x-3">
                          <InfoCircleOutlined className="text-blue-500 mt-0.5" />
                          <div>
                            <p className="text-sm font-medium text-blue-900">Dynamic Schema</p>
                            <p className="text-xs text-blue-700 mt-1">This tool uses a dynamic input schema.</p>
                          </div>
                        </div>
                      </div>
                    )}
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
                  <div className={`text-center ${result || error ? 'py-4' : 'py-12'} bg-gray-50 rounded-lg border border-gray-200`}>
                    <div className="max-w-sm mx-auto">
                      {!(result || error) && (
                        <div className="mb-4">
                          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </div>
                      )}
                      <h4 className="text-sm font-medium text-gray-900 mb-1">No Parameters Required</h4>
                      <p className="text-xs text-gray-500">This tool can be called without any input parameters.</p>
                    </div>
                  </div>
                ) : (
                  <div className={`${result || error ? 'space-y-2' : 'space-y-4'}`}>
                    {Object.entries(schema.properties).map(([key, prop]) => (
                      <Form.Item
                        key={key}
                        label={
                          <span className="text-sm font-medium text-gray-700 flex items-center">
                            {key}{" "}
                            {schema.required?.includes(key) && <span className="text-red-500">*</span>}
                            {prop.description && !result && !error && (
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
                            className="w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm transition-colors"
                          />
                        )}

                        {prop.type === "boolean" && (
                          <div className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                            <input
                              type="checkbox"
                              className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded transition-colors"
                            />
                            <span className="text-sm text-gray-700">Enable this option</span>
                          </div>
                        )}
                      </Form.Item>
                    ))}
                  </div>
                )}

                <div className={`${result || error ? 'pt-3 border-t border-gray-100' : 'pt-6 border-t border-gray-100'}`}>
                  <Button
                    onClick={() => form.submit()}
                    disabled={isLoading}
                    variant="primary"
                    className="w-full"
                    loading={isLoading}
                  >
                    {isLoading ? "Calling Tool..." : (result || error) ? "Call Again" : "Call Tool"}
                  </Button>
                </div>
              </Form>
            </div>
          </div>

          {/* Result Section */}
          {!result && !error && !isLoading && (
            <div className="space-y-6">
              <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
                <div className="border-b border-gray-100 px-6 py-4">
                  <h3 className="text-lg font-semibold text-gray-900">Tool Result</h3>
                </div>
                <div className="p-6">
                  <div className="flex flex-col justify-center items-center h-64 text-gray-500">
                    <div className="text-center max-w-sm">
                      <div className="mb-4">
                        <svg className="mx-auto h-16 w-16 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                      </div>
                      <h4 className="text-sm font-medium text-gray-900 mb-2">Ready to Call Tool</h4>
                      <p className="text-xs text-gray-500 leading-relaxed">
                        Configure the input parameters on the left and click "Call Tool" to see the results here.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Results when present - full width */}
          {(result || error || isLoading) && (
            <div className="bg-white border border-gray-200 rounded-lg shadow-sm">
              <div className="border-b border-gray-100 px-6 py-4">
                <h3 className="text-lg font-semibold text-gray-900">Tool Result</h3>
              </div>
              
              <div className="p-6">
                {/* Result Control Bar */}
                {result && !isLoading && !error && (
                  <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <svg className="h-5 w-5 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <h4 className="text-sm font-medium text-green-900">Tool executed successfully</h4>
                        {duration !== null && (
                          <span className="text-xs text-green-600 ml-2">
                            • {(duration / 1000).toFixed(2)}s
                          </span>
                        )}
                      </div>

                      <div className="flex items-center space-x-2">
                        <div className="flex bg-white rounded-lg border border-green-300 p-1">
                          <button
                            onClick={() => setViewMode('formatted')}
                            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                              viewMode === 'formatted'
                                ? 'bg-green-100 text-green-800'
                                : 'text-green-600 hover:text-green-800'
                            }`}
                          >
                            Formatted
                          </button>
                          <button
                            onClick={() => setViewMode('json')}
                            className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${
                              viewMode === 'json'
                                ? 'bg-green-100 text-green-800'
                                : 'text-green-600 hover:text-green-800'
                            }`}
                          >
                            JSON
                          </button>
                        </div>
                        
                        <button 
                          onClick={handleCopyResult}
                          className="p-1 hover:bg-green-100 rounded text-green-700"
                          title="Copy response"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                          </svg>
                        </button>
                      </div>
                    </div>
                  </div>
                )}



                <div className="min-h-[400px] max-h-[500px] overflow-y-auto">
                  {isLoading && (
                    <div className="flex flex-col justify-center items-center h-full py-16 text-gray-500">
                      <div className="relative">
                        <div className="animate-spin rounded-full h-10 w-10 border-2 border-gray-200"></div>
                        <div className="animate-spin rounded-full h-10 w-10 border-2 border-blue-600 border-t-transparent absolute top-0"></div>
                      </div>
                      <p className="text-sm font-medium mt-4">Calling tool...</p>
                      <p className="text-xs text-gray-400 mt-1">Please wait while we process your request</p>
                    </div>
                  )}

                  {error && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                      <div className="flex items-start space-x-3">
                        <div className="flex-shrink-0">
                          <svg className="h-5 w-5 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-2">
                            <h4 className="text-sm font-medium text-red-900">Tool Call Failed</h4>
                            {duration !== null && (
                              <span className="text-xs text-red-600">
                                • {(duration / 1000).toFixed(2)}s
                              </span>
                            )}
                          </div>
                          <div className="bg-white border border-red-200 rounded-lg p-3 max-h-64 overflow-y-auto">
                            <pre className="text-xs whitespace-pre-wrap text-red-700 font-mono">{error.message}</pre>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {result && !isLoading && !error && (
                    <div className="space-y-4">
                      {viewMode === 'formatted' ? (
                        // Formatted View
                        result.map((content: any, idx: number) => (
                          <div key={idx} className="border border-gray-200 rounded-lg overflow-hidden">
                            {content.type === "text" && (
                              <div>
                                <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
                                  <span className="text-xs font-medium text-gray-700 uppercase tracking-wide">Text Response</span>
                                </div>
                                <div className="p-4">
                                  <div className="bg-white rounded-lg border border-gray-200 max-h-80 overflow-y-auto">
                                    <div className="p-4 space-y-3">
                                      {content.text.split('\n\n').map((section: string, sectionIndex: number) => {
                                        if (section.trim() === '') return null;
                                        
                                        // Handle headers (## or ###)
                                        if (section.startsWith('##')) {
                                          const headerText = section.replace(/^#+\s/, '');
                                          return (
                                            <div key={sectionIndex} className="border-b border-gray-200 pb-2 mb-3">
                                              <h3 className="text-base font-semibold text-gray-900">{headerText}</h3>
                                            </div>
                                          );
                                        }
                                        
                                        // Handle URL-containing sections
                                        const urlRegex = /(https?:\/\/[^\s\)]+)/g;
                                        if (urlRegex.test(section)) {
                                          const parts = section.split(urlRegex);
                                          return (
                                            <div key={sectionIndex} className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                                              <div className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap">
                                                {parts.map((part, partIndex) => {
                                                  if (urlRegex.test(part)) {
                                                    return (
                                                      <a
                                                        key={partIndex}
                                                        href={part}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="text-blue-600 hover:text-blue-800 underline break-all"
                                                      >
                                                        {part}
                                                      </a>
                                                    );
                                                  }
                                                  return part;
                                                })}
                                              </div>
                                            </div>
                                          );
                                        }
                                        
                                        // Handle score information
                                        if (section.includes('Score:')) {
                                          return (
                                            <div key={sectionIndex} className="bg-green-50 border-l-4 border-green-400 p-3 rounded-r">
                                              <p className="text-sm text-green-800 font-medium whitespace-pre-wrap">{section}</p>
                                            </div>
                                          );
                                        }
                                        
                                        // Regular content sections
                                        return (
                                          <div key={sectionIndex} className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                                            <div className="text-sm text-gray-700 leading-relaxed whitespace-pre-wrap font-mono">
                                              {section}
                                            </div>
                                          </div>
                                        );
                                      }).filter(Boolean)}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            )}

                            {content.type === "image" && content.url && (
                              <div>
                                <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
                                  <span className="text-xs font-medium text-gray-700 uppercase tracking-wide">Image Response</span>
                                </div>
                                <div className="p-4">
                                  <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                                    <img
                                      src={content.url}
                                      alt="Tool result"
                                      className="max-w-full h-auto rounded-lg shadow-sm"
                                    />
                                  </div>
                                </div>
                              </div>
                            )}

                            {content.type === "embedded_resource" && (
                              <div>
                                <div className="bg-gray-50 px-4 py-2 border-b border-gray-200">
                                  <span className="text-xs font-medium text-gray-700 uppercase tracking-wide">Embedded Resource</span>
                                </div>
                                <div className="p-4">
                                  <div className="flex items-center space-x-3 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                                    <div className="flex-shrink-0">
                                      <svg className="h-6 w-6 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                      </svg>
                                    </div>
                                    <div className="flex-1">
                                      <p className="text-sm font-medium text-blue-900">Resource Type: {content.resource_type}</p>
                                      {content.url && (
                                        <a
                                          href={content.url}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          className="inline-flex items-center text-sm text-blue-600 hover:text-blue-800 hover:underline mt-1 transition-colors"
                                        >
                                          View Resource
                                          <svg className="ml-1 h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
                                            <path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z" />
                                            <path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z" />
                                          </svg>
                                        </a>
                                      )}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        ))
                      ) : (
                        // JSON View
                        <div className="bg-white rounded-lg border border-gray-200">
                          <div className="p-4 overflow-auto max-h-96 bg-gray-50">
                            <pre className="text-xs font-mono whitespace-pre-wrap break-all text-gray-800">
                              {JSON.stringify(result, null, 2)}
                            </pre>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </Modal>
  );
} 