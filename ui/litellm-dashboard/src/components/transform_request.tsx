import React, { useState } from 'react';
import { Button, Select, Tabs, message } from 'antd';
import { CopyOutlined } from '@ant-design/icons';

interface TransformRequestPanelProps {
  accessToken: string | null;
}

const TransformRequestPanel: React.FC<TransformRequestPanelProps> = ({ accessToken }) => {
  const [originalRequestJSON, setOriginalRequestJSON] = useState(`{
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
}`);
  
  const [transformedResponse, setTransformedResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // Function to format the curl command with proper indentation
  const formatCurlCommand = (response: string) => {
    // Remove the "POST Request Sent from LiteLLM:" prefix
    let cleaned = response.replace("\n\nPOST Request Sent from LiteLLM:\n", "");
    
    // Format curl command
    try {
        // Extract URL
        const urlMatch = cleaned.match(/https:\/\/[^\s\\]+/);
        const url = urlMatch ? urlMatch[0] : "api.openai.com/v1/";
        
        // Extract data payload
        console.log(`cleaned: ${cleaned}`);
        // Updated regex to better handle the -d parameter with single quotes
        const dataMatch = cleaned.match(/-d\s+'([^']+)'/);
        let dataPayload = "{}";
        
        if (dataMatch && dataMatch[1]) {
          console.log(`dataMatch[1]: ${dataMatch[1]}`);
          // Format the JSON part
          try {
            // Replace single quotes with double quotes, but we need to be careful about nested quotes
            // First, let's convert the Python-style single-quoted JSON to valid JSON
            const processedStr = dataMatch[1]
              .replace(/'/g, '"')           // Replace all single quotes with double quotes
              .replace(/"\s*:\s*"/g, '":"') // Fix spacing in key-value pairs
              .replace(/"\s*,\s*"/g, '","') // Fix spacing in arrays
              .replace(/{\s*"/g, '{"')      // Fix spacing at start of objects
              .replace(/"\s*}/g, '"}');     // Fix spacing at end of objects
            
            const jsonObj = JSON.parse(processedStr);
            dataPayload = JSON.stringify(jsonObj, null, 2);
          } catch (e) {
            console.error("Error parsing JSON:", e);
            // If parsing fails, at least return the raw data
            dataPayload = dataMatch[1];
          }
        }
      } catch (e) {
      // If formatting fails, return a basic cleanup
      return cleaned
        .replace(/\\\n/g, ' \\\n  ') // Add consistent indentation
        .replace(/-d '/, "-d '\n  ") // Format the data part
        .replace(/}'$/, "}\n'"); // Close data block nicely
    }
  };
  
  // Function to handle the transform request using fetch
  const handleTransform = async () => {
    setIsLoading(true);
    
    try {
      // Parse the JSON from the textarea
      let requestBody;
      try {
        requestBody = JSON.parse(originalRequestJSON);
      } catch (e) {
        message.error('Invalid JSON in request body');
        setIsLoading(false);
        return;
      }
      
      // Create the request payload
      const payload = {
        call_type: "completion",
        request_body: requestBody
      };
      
      // Make the API call using fetch
      const response = await fetch('http://0.0.0.0:4000/utils/transform_request', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      
      // Get the response as text
      const rawText = await response.text();
      
      // Format the curl command
      const formattedCurl = formatCurlCommand(rawText);
      
      // Update the transformed response state
      setTransformedResponse(formattedCurl);
      message.success('Request transformed successfully');
    } catch (err) {
      console.error('Error transforming request:', err);
      message.error('Failed to transform request');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ 
      display: 'flex', 
      gap: '16px', 
      width: '100%', 
      height: '100%', 
      overflow: 'hidden'
    }}>
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
          onChange={(e) => setOriginalRequestJSON(e.target.value)}
        />
        
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between',
          marginTop: 'auto'
        }}>
          <Select 
            defaultValue="OpenAI" 
            style={{ width: '120px' }}
          >
            <Select.Option value="OpenAI">OpenAI</Select.Option>
          </Select>
          
          <Button 
            type="primary" 
            style={{ 
              backgroundColor: '#000', 
              display: 'flex', 
              alignItems: 'center', 
              gap: '8px'
            }}
            onClick={handleTransform}
            loading={isLoading}
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
        <div style={{ marginBottom: '24px' }}>
          <h2 style={{ fontSize: '24px', fontWeight: 'bold', margin: '0 0 4px 0' }}>Transformed Request</h2>
          <p style={{ color: '#666', margin: 0 }}>How LiteLLM transforms your request for OpenAI</p>
        </div>
        
        <Tabs defaultActiveKey="JSON" style={{ marginBottom: '16px' }}>
          <Tabs.TabPane tab="JSON" key="JSON" />
          <Tabs.TabPane tab="Differences" key="Differences" />
        </Tabs>
        
        <div style={{ 
          position: 'relative',
          backgroundColor: '#f5f5f5',
          borderRadius: '6px',
          flex: '1 1 auto',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}>
          <pre 
            style={{
              padding: '16px',
              fontFamily: 'monospace',
              fontSize: '14px',
              margin: 0,
              overflow: 'auto',
              flex: '1 1 auto'
            }}
          >
            {transformedResponse || `curl -X POST \\
  https://api.openai.com/v1/chat/completions \\
  -H 'Authorization: *****' \\
  -H 'Content-Type: application/json' \\
  -d '{
  "model": "gpt-4o",
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
}'`}
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
            onClick={() => {
              navigator.clipboard.writeText(transformedResponse || '');
              message.success('Copied to clipboard');
            }}
          />
        </div>
      </div>
    </div>
  );
};

export default TransformRequestPanel;