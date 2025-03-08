import React from 'react';
import { Button, Select, Tabs } from 'antd';
import { CopyOutlined } from '@ant-design/icons';

interface TransformRequestPanelProps {
  accessToken: string | null;
}

const TransformRequestPanel: React.FC<TransformRequestPanelProps> = ({ accessToken }) => {
  // Original request content with truncated display to match screenshot
  const originalRequestJSON = `{
  "model": "openai/gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Explain quantum computing in simple terms"
    }
  ],
}`;

  // Full transformed request content
  const transformedRequestJSON = `{
  "model": "openai/gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Explain quantum computing in simple terms"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 500,
  "stream": true
}`;

return (
    <div style={{ 
      display: 'flex', 
      gap: '16px', 
      width: '100%', 
      height: '100%', 
      overflow: 'hidden'
    }}
    className='mt-4'>
      {/* Original Request Panel */}
      <div style={{ 
        flex: '1 1 50%',
        display: 'flex', 
        flexDirection: 'column',
        border: '1px solid #e8e8e8', 
        borderRadius: '8px', 
        padding: '24px',
        overflow: 'hidden'
      }}>
        <div style={{ marginBottom: '24px' }}>
          <h2 style={{ fontSize: '24px', fontWeight: 'bold', margin: '0 0 4px 0' }}>Original Request</h2>
          <p style={{ color: '#666', margin: 0 }}>The request you would send to LiteLLM</p>
        </div>
        
        <textarea 
          style={{ 
            flex: '1 1 auto',
            width: '100%', 
            minHeight: '240px',
            padding: '16px', 
            border: '1px solid #e8e8e8', 
            borderRadius: '6px',
            fontFamily: 'monospace',
            fontSize: '14px',
            resize: 'none',
            marginBottom: '24px',
            overflow: 'auto'
          }}
          value={originalRequestJSON}
          readOnly
        />
        
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between',
          marginTop: 'auto'
        }}>
          <Button 
            type="primary" 
            style={{ 
              backgroundColor: '#000', 
              display: 'flex', 
              alignItems: 'center', 
              gap: '8px'
            }}
          >
            <span>Transform</span>
            <span>→</span>
          </Button>
        </div>
      </div>
      
      {/* Transformed Request Panel */}
      <div style={{ 
        flex: '1 1 50%',
        display: 'flex',
        flexDirection: 'column',
        border: '1px solid #e8e8e8', 
        borderRadius: '8px', 
        padding: '24px',
        overflow: 'hidden'
      }}>
        <h2 style={{ fontSize: '24px', fontWeight: 'bold', margin: '0 0 4px 0' }}>Transformed Request</h2>
        <p style={{ color: '#666', marginBottom: '24px' }}>How LiteLLM transforms your request for OpenAI</p>
        
        <Tabs defaultActiveKey="JSON" style={{ marginBottom: '16px' }}>
          <Tabs.TabPane tab="JSON" key="JSON" />
          <Tabs.TabPane tab="Differences" key="Differences" />
        </Tabs>
        
        <div style={{ 
          position: 'relative',
          backgroundColor: '#f5f5f5',
          borderRadius: '6px',
          minHeight: '300px'
        }}>
          <pre style={{
            padding: '16px',
            fontFamily: 'monospace',
            fontSize: '14px',
            whiteSpace: 'pre',
            margin: 0,
            // overflow: 'auto',
            maxHeight: '400px'
          }}>
            {transformedRequestJSON}
          </pre>
          
          <Button 
            type="text" 
            icon={<CopyOutlined />} 
            style={{
              position: 'absolute',
              right: '8px',
              top: '8px'
            }}
            size="small"
          />
        </div>
      </div>
    </div>
  );
};

export default TransformRequestPanel;