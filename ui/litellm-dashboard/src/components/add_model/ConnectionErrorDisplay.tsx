import React from 'react';
import { Typography, Space, Button, Divider } from 'antd';
import { WarningOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { ErrorViewer } from '../view_logs/ErrorViewer';

const { Text } = Typography;

interface ConnectionErrorDisplayProps {
  error: Error | string;
  modelName?: string;
  onClose?: () => void;
}

const ConnectionErrorDisplay: React.FC<ConnectionErrorDisplayProps> = ({ 
  error, 
  modelName = "this model",
  onClose 
}) => {
  const errorMessage = typeof error === 'string' ? error : error.message;
  
  // Create an error info object compatible with ErrorViewer
  const errorInfo = {
    error_message: errorMessage.split('\n')[0],
    traceback: errorMessage,
    // We don't have these fields from the connection test error, but the component handles undefined
    error_class: undefined,
    llm_provider: undefined,
    error_code: undefined
  };

  return (
      
      <div style={{ padding: '16px 24px' }}>

        
        {/* Use the ErrorViewer component for consistent error display */}
        <ErrorViewer errorInfo={errorInfo} />
        
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