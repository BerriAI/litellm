import React from 'react';
import { Typography, Space, Button, Divider, message } from 'antd';
import { WarningOutlined, InfoCircleOutlined, CopyOutlined } from '@ant-design/icons';
import { testConnectionRequest } from "../networking";
import { prepareModelAddRequest } from "./handle_add_model_submit";

const { Text } = Typography;

interface ConnectionErrorDisplayProps {
  formValues: Record<string, any>;
  accessToken: string;
  testMode: string;
  modelName?: string;
  onClose?: () => void;
}

const ConnectionErrorDisplay: React.FC<ConnectionErrorDisplayProps> = ({ 
  formValues, 
  accessToken, 
  testMode, 
  modelName = "this model", 
  onClose 
}) => {
  const [error, setError] = React.useState<Error | string | null>(null);
  const [rawRequest, setRawRequest] = React.useState<any>(null);
  const [rawResponse, setRawResponse] = React.useState<any>(null);
  const [isLoading, setIsLoading] = React.useState<boolean>(true);
  const [isSuccess, setIsSuccess] = React.useState<boolean>(false);

  const testModelConnection = async () => {
    setIsLoading(true);
    try {
      const result = await prepareModelAddRequest(formValues, accessToken, null);
      if (!result) throw new Error("Failed to prepare model data");

      const { litellmParamsObj } = result;
      const requestBody = { ...litellmParamsObj, mode: testMode };

      const response = await testConnectionRequest(accessToken, requestBody);
      if (response.status === "success") {
        message.success("Connection test successful!");
        setError(null);
        setIsSuccess(true);
      } else {
        const errorMessage = response.result?.error || response.message || "Unknown error";
        setError(errorMessage);
        setRawRequest(requestBody);
        setRawResponse(response.result?.raw_request_typed_dict);
        setIsSuccess(false);
      }
    } catch (error) {
      console.error("Test connection error:", error);
      setError(error instanceof Error ? error.message : String(error));
      setIsSuccess(false);
    } finally {
      setIsLoading(false);
    }
  };

  React.useEffect(() => {
    testModelConnection();
  }, []);

  const errorMessage = typeof error === 'string' ? error : error?.message;

  const formatCurlCommand = (apiBase: string, requestBody: Record<string, any>, requestHeaders: Record<string, string>) => {
    const formattedBody = JSON.stringify(requestBody, null, 2)
      .split('\n')
      .map(line => `  ${line}`)
      .join('\n');

    const headerString = Object.entries(requestHeaders)
      .map(([key, value]) => `-H '${key}: ${value}'`)
      .join(' \\\n  ');

    return `curl -X POST \\
  ${apiBase} \\
  ${headerString ? `${headerString} \\\n  ` : ''}-H 'Content-Type: application/json' \\
  -d '{
${formattedBody}
  }'`;
  };

  const curlCommand = rawResponse ? formatCurlCommand(
    rawResponse.raw_request_api_base,
    rawResponse.raw_request_body,
    rawResponse.raw_request_headers || {}
  ) : '';

  return (
    <div style={{ padding: '16px 24px' }}>
      {isLoading ? (
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <div>Testing connection to {modelName}...</div>
          {/* You could add a spinner here */}
        </div>
      ) : isSuccess ? (
        <div style={{ textAlign: 'center', padding: '20px' }}>
          <Text type="success" style={{ fontSize: '16px' }}>
            Connection to {modelName} successful!
          </Text>
        </div>
      ) : (
        <>
          <div>
            <Text type="danger" style={{ fontSize: '16px' }}>Connection to {modelName} failed</Text>
            <Divider style={{ margin: '16px 0' }} />
            <Text type="danger">{errorMessage}</Text>
            <Divider style={{ margin: '16px 0' }} />
            <div>
              <h3>Raw Request</h3>
              <pre style={{ backgroundColor: '#f5f5f5', padding: '16px', borderRadius: '6px' }}>
                {curlCommand || "No request data"}
              </pre>
              <Button 
                type="text" 
                icon={<CopyOutlined />} 
                onClick={() => {
                  navigator.clipboard.writeText(curlCommand || '');
                  message.success('Copied to clipboard');
                }}
              >
                Copy to Clipboard
              </Button>
            </div>
          </div>
        </>
      )}
      <Divider style={{ margin: '16px 0' }} />
      <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
        <Button 
          type="link" 
          href="https://docs.litellm.ai/docs/providers" 
          target="_blank"
          icon={<InfoCircleOutlined />}
        >
          View Documentation
        </Button>
      </div>
    </div>
  );
};

export default ConnectionErrorDisplay; 