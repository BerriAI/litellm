import React from 'react';
import { Typography, Space, Button, Divider, message } from 'antd';
import { WarningOutlined, InfoCircleOutlined, CopyOutlined } from '@ant-design/icons';
import { testMCPConnectionRequest } from "./networking";
import NotificationsManager from "./molecules/notifications_manager";

const { Text } = Typography;

interface MCPConnectionTestProps {
  formValues: Record<string, any>;
  accessToken: string;
  serverName?: string;
  onClose?: () => void;
  onTestComplete?: () => void;
}

const MCPConnectionTest: React.FC<MCPConnectionTestProps> = ({ 
  formValues, 
  accessToken, 
  serverName = "this MCP server", 
  onClose,
  onTestComplete
}) => {
  const [connectionError, setConnectionError] = React.useState<Error | string | null>(null);
  const [rawRequest, setRawRequest] = React.useState<any>(null);
  const [rawResponse, setRawResponse] = React.useState<any>(null);
  const [isLoading, setIsLoading] = React.useState<boolean>(true);
  const [connectionSuccess, setConnectionSuccess] = React.useState<boolean>(false);
  const [showDetails, setShowDetails] = React.useState<boolean>(false);

  const testMCPConnection = async () => {
    setIsLoading(true);
    setShowDetails(false);
    setConnectionError(null);
    setRawRequest(null);
    setRawResponse(null);
    setConnectionSuccess(false);
    
    // Add a small delay to ensure form values are fully populated
    await new Promise(resolve => setTimeout(resolve, 100));
    
    try {
      console.log("Testing MCP connection with form values:", formValues);
      
      // Prepare the MCP server config from form values
      const mcpServerConfig = {
        server_id: formValues.server_id || "",
        alias: formValues.alias || "",
        url: formValues.url,
        transport: formValues.transport,
        auth_type: formValues.auth_type,
        mcp_info: formValues.mcp_info,
      };

      setRawRequest(mcpServerConfig);

      // Test connection
      const connectionResponse = await testMCPConnectionRequest(accessToken, mcpServerConfig);
      console.log("Connection test response:", connectionResponse);
      
      if (connectionResponse.status === "ok") {
        setConnectionError(null);
        setConnectionSuccess(true);
      } else {
        const errorMessage = connectionResponse.message || "Unknown connection error";
        setConnectionError(errorMessage);
        setRawResponse(connectionResponse);
      }
    } catch (error) {
      console.error("MCP connection test error:", error);
      setConnectionError(error instanceof Error ? error.message : String(error));
    } finally {
      setIsLoading(false);
      if (onTestComplete) onTestComplete();
    }
  };

  React.useEffect(() => {
    // Run the test once when component mounts
    // Add a small timeout to ensure form values are ready
    const timer = setTimeout(() => {
      testMCPConnection();
    }, 200);
    
    return () => clearTimeout(timer);
  }, []); // Empty dependency array means this runs once on mount

  const getCleanErrorMessage = (errorMsg: string) => {
    if (!errorMsg) return "Unknown error";
    
    const mainError = errorMsg.split('stack trace:')[0].trim();
    
    const cleanedError = mainError.replace(/^(.*?)Error: /, '');
    
    return cleanedError;
  };

  const connectionErrorMessage = typeof connectionError === 'string' 
    ? getCleanErrorMessage(connectionError) 
    : connectionError?.message ? getCleanErrorMessage(connectionError.message) : "Unknown error";

  const formatMCPRequest = (mcpConfig: Record<string, any>) => {
    return JSON.stringify(mcpConfig, null, 2);
  };

  const isOverallSuccess = connectionSuccess && !connectionError;

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
          <Text style={{ fontSize: '16px' }}>Testing connection to {serverName}...</Text>
          <style jsx>{`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}</style>
        </div>
      ) : isOverallSuccess ? (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '32px 20px' }}>
            <div style={{ color: '#52c41a', fontSize: '24px', display: 'flex', alignItems: 'center' }}>
              <svg viewBox="64 64 896 896" focusable="false" data-icon="check-circle" width="1em" height="1em" fill="currentColor" aria-hidden="true">
                <path d="M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm193.5 301.7l-210.6 292a31.8 31.8 0 01-51.7 0L318.5 484.9c-3.8-5.3 0-12.7 6.5-12.7h46.9c10.2 0 19.9 4.9 25.9 13.3l71.2 98.8 157.2-218c6-8.3 15.6-13.3 25.9-13.3H699c6.5 0 10.3 7.4 6.5 12.7z"></path>
              </svg>
            </div>
            <Text type="success" style={{ fontSize: '18px', fontWeight: 500, marginLeft: '10px' }}>
              Connection to {serverName} successful!
            </Text>
          </div>
          

        </div>
      ) : (
        <>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '20px' }}>
              <WarningOutlined style={{ color: '#ff4d4f', fontSize: '24px', marginRight: '12px' }} />
              <Text type="danger" style={{ fontSize: '18px', fontWeight: 500 }}>Connection to {serverName} failed</Text>
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
              <Text type="danger" style={{ fontSize: '14px', lineHeight: '1.5' }}>{connectionErrorMessage}</Text>
              
              {connectionError && (
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
                  {typeof connectionError === 'string' ? connectionError : JSON.stringify(connectionError, null, 2)}
                </pre>
              </div>
            )}
            
            <div>
              <Text strong style={{ display: 'block', marginBottom: '8px', fontSize: '15px' }}>MCP Server Configuration</Text>
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
                {formatMCPRequest(rawRequest || {})}
              </pre>
              <Button 
                style={{ marginTop: '8px' }}
                icon={<CopyOutlined />} 
                onClick={() => {
                  navigator.clipboard.writeText(formatMCPRequest(rawRequest || {}));
                  NotificationsManager.success('Copied to clipboard');
                }}
              >
                Copy Configuration
              </Button>
            </div>
          </div>
        </>
      )}
      <Divider style={{ margin: '24px 0 16px' }} />
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Button 
          type="link" 
          href="https://docs.litellm.ai/docs/proxy/mcp_server" 
          target="_blank"
          icon={<InfoCircleOutlined />}
        >
          View MCP Documentation
        </Button>
        <Space>
          <Button 
            onClick={testMCPConnection}
            loading={isLoading}
          >
            Test Again
          </Button>
          <Button 
            type="primary"
            onClick={onClose}
          >
            Close
          </Button>
        </Space>
      </div>
    </div>
  );
};

export default MCPConnectionTest; 