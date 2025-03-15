import React from 'react';
import { Typography, Space, Button, Divider, message } from 'antd';
import { WarningOutlined, InfoCircleOutlined, CopyOutlined } from '@ant-design/icons';
import { testConnectionRequest } from "../networking";
import { prepareModelAddRequest } from "./handle_add_model_submit";

const { Text } = Typography;

interface ModelConnectionTestProps {
  formValues: Record<string, any>;
  accessToken: string;
  testMode: string;
  modelName?: string;
  onClose?: () => void;
  onTestComplete?: () => void;
}

const ModelConnectionTest: React.FC<ModelConnectionTestProps> = ({ 
  formValues, 
  accessToken, 
  testMode, 
  modelName = "this model", 
  onClose,
  onTestComplete
}) => {
  const [error, setError] = React.useState<Error | string | null>(null);
  const [rawRequest, setRawRequest] = React.useState<any>(null);
  const [rawResponse, setRawResponse] = React.useState<any>(null);
  const [isLoading, setIsLoading] = React.useState<boolean>(true);
  const [isSuccess, setIsSuccess] = React.useState<boolean>(false);
  const [showDetails, setShowDetails] = React.useState<boolean>(false);

  const testModelConnection = async () => {
    setIsLoading(true);
    setShowDetails(false);
    setError(null);
    setRawRequest(null);
    setRawResponse(null);
    setIsSuccess(false);
    
    // Add a small delay to ensure form values are fully populated
    await new Promise(resolve => setTimeout(resolve, 100));
    
    try {
      console.log("Testing connection with form values:", formValues);
      const result = await prepareModelAddRequest(formValues, accessToken, null);
      
      if (!result) {
        console.log("No result from prepareModelAddRequest");
        setError("Failed to prepare model data. Please check your form inputs.");
        setIsSuccess(false);
        setIsLoading(false);
        return;
      }

      console.log("Result from prepareModelAddRequest:", result);

      const { litellmParamsObj, modelInfoObj, modelName: returnedModelName } = result;

      const response = await testConnectionRequest(accessToken, litellmParamsObj, modelInfoObj?.mode);
      if (response.status === "success") {
        message.success("Connection test successful!");
        setError(null);
        setIsSuccess(true);
      } else {
        const errorMessage = response.result?.error || response.message || "Unknown error";
        setError(errorMessage);
        setRawRequest(litellmParamsObj);
        setRawResponse(response.result?.raw_request_typed_dict);
        setIsSuccess(false);
      }
    } catch (error) {
      console.error("Test connection error:", error);
      setError(error instanceof Error ? error.message : String(error));
      setIsSuccess(false);
    } finally {
      setIsLoading(false);
      if (onTestComplete) onTestComplete();
    }
  };

  React.useEffect(() => {
    // Run the test once when component mounts
    // Add a small timeout to ensure form values are ready
    const timer = setTimeout(() => {
      testModelConnection();
    }, 200);
    
    return () => clearTimeout(timer);
  }, []); // Empty dependency array means this runs once on mount

  const getCleanErrorMessage = (errorMsg: string) => {
    if (!errorMsg) return "Unknown error";
    
    const mainError = errorMsg.split('stack trace:')[0].trim();
    
    const cleanedError = mainError.replace(/^litellm\.(.*?)Error: /, '');
    
    return cleanedError;
  };

  const errorMessage = typeof error === 'string' 
    ? getCleanErrorMessage(error) 
    : error?.message ? getCleanErrorMessage(error.message) : "Unknown error";

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
    <div style={{ padding: '24px', borderRadius: '8px', backgroundColor: '#fff' }}>
      {isLoading ? (
        <div style={{ textAlign: 'center', padding: '32px 20px' }}>
          <div className="loading-spinner" style={{ marginBottom: '16px' }}>
            {/* Simple CSS spinner */}
            <div style={{ 
              border: '3px solid #f3f3f3',
              borderTop: '3px solid #1890ff',
              borderRadius: '50%',
              width: '30px',
              height: '30px',
              animation: 'spin 1s linear infinite',
              margin: '0 auto'
            }} />
          </div>
          <Text style={{ fontSize: '16px' }}>Testing connection to {modelName}...</Text>
          <style jsx>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}</style>
        </div>
      ) : isSuccess ? (
        <div style={{ textAlign: 'center', padding: '32px 20px' }}>
          <div style={{ color: '#52c41a', fontSize: '32px', marginBottom: '16px' }}>
            <svg viewBox="64 64 896 896" focusable="false" data-icon="check-circle" width="1em" height="1em" fill="currentColor" aria-hidden="true">
              <path d="M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm193.5 301.7l-210.6 292a31.8 31.8 0 01-51.7 0L318.5 484.9c-3.8-5.3 0-12.7 6.5-12.7h46.9c10.2 0 19.9 4.9 25.9 13.3l71.2 98.8 157.2-218c6-8.3 15.6-13.3 25.9-13.3H699c6.5 0 10.3 7.4 6.5 12.7z"></path>
            </svg>
          </div>
          <Text type="success" style={{ fontSize: '18px', fontWeight: 500 }}>
            Connection to {modelName} successful!
          </Text>
        </div>
      ) : (
        <>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
              <WarningOutlined style={{ color: '#ff4d4f', fontSize: '24px', marginRight: '12px' }} />
              <Text type="danger" style={{ fontSize: '18px', fontWeight: 500 }}>Connection to {modelName} failed</Text>
            </div>
            
            <div style={{ 
              backgroundColor: '#fff2f0', 
              border: '1px solid #ffccc7', 
              borderRadius: '8px', 
              padding: '16px', 
              marginBottom: '20px',
              boxShadow: '0 1px 2px rgba(0, 0, 0, 0.03)'
            }}>
              <Text strong style={{ display: 'block', marginBottom: '8px' }}>Error: </Text>
              <Text type="danger" style={{ fontSize: '14px', lineHeight: '1.5' }}>{errorMessage}</Text>
              
              {error && (
                <div style={{ marginTop: '12px' }}>
                  <Button 
                    type="link" 
                    onClick={() => setShowDetails(!showDetails)}
                    style={{ paddingLeft: 0, height: 'auto' }}
                  >
                    {showDetails ? 'Hide Details' : 'Show Details'}
                  </Button>
                </div>
              )}
            </div>
            
            {showDetails && (
              <div style={{ marginBottom: '20px' }}>
                <Text strong style={{ display: 'block', marginBottom: '8px', fontSize: '15px' }}>Troubleshooting Details</Text>
                <pre style={{ 
                  backgroundColor: '#f5f5f5', 
                  padding: '16px', 
                  borderRadius: '8px',
                  fontSize: '13px',
                  maxHeight: '200px',
                  overflow: 'auto',
                  border: '1px solid #e8e8e8',
                  lineHeight: '1.5'
                }}>
                  {typeof error === 'string' ? error : JSON.stringify(error, null, 2)}
                </pre>
              </div>
            )}
            
            <div>
              <Text strong style={{ display: 'block', marginBottom: '8px', fontSize: '15px' }}>API Request</Text>
              <pre style={{ 
                backgroundColor: '#f5f5f5', 
                padding: '16px', 
                borderRadius: '8px',
                fontSize: '13px',
                maxHeight: '250px',
                overflow: 'auto',
                border: '1px solid #e8e8e8',
                lineHeight: '1.5'
              }}>
                {curlCommand || "No request data available"}
              </pre>
              <Button 
                style={{ marginTop: '8px' }}
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
      <Divider style={{ margin: '24px 0 16px' }} />
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
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

export default ModelConnectionTest; 