import React, { useState } from 'react';
import { Button, message } from 'antd';
import { CopyOutlined } from '@ant-design/icons';
import styles from './transform_request.module.css';

interface TransformRequestPanelProps {
  accessToken: string | null;
}

interface TransformResponse {
  raw_request_api_base: string;
  raw_request_body: Record<string, any>;
  raw_request_headers: Record<string, string>;
}

const TransformRequestPanel: React.FC<TransformRequestPanelProps> = ({ accessToken }) => {
  const [originalRequestJSON, setOriginalRequestJSON] = useState(`{
  "model": "bedrock/gpt-4o",
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
  
  // Function to format curl command from API response parts
  const formatCurlCommand = (apiBase: string, requestBody: Record<string, any>, requestHeaders: Record<string, string>) => {
    // Format the request body as nicely indented JSON with 2 spaces
    const formattedBody = JSON.stringify(requestBody, null, 2)
      // Add additional indentation for the entire body
      .split('\n')
      .map(line => `  ${line}`)
      .join('\n');

    // Build headers string with consistent indentation
    const headerString = Object.entries(requestHeaders)
      .map(([key, value]) => `-H '${key}: ${value}'`)
      .join(' \\\n  ');

    // Build the curl command with consistent indentation
    return `curl -X POST \\
  ${apiBase} \\
  ${headerString ? `${headerString} \\\n  ` : ''}-H 'Content-Type: application/json' \\
  -d '{
${formattedBody}
  }'`;
  };
  
  // Function to handle the transform request
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
      
      // Parse the response as JSON
      const data = await response.json();
      console.log("API response:", data);
      
      // Check if the response has the expected fields
      if (data.raw_request_api_base && data.raw_request_body) {
        // Format the curl command with the separate parts
        const formattedCurl = formatCurlCommand(
          data.raw_request_api_base, 
          data.raw_request_body, 
          data.raw_request_headers || {}
        );
        
        // Update state with the formatted curl command
        setTransformedResponse(formattedCurl);
        message.success('Request transformed successfully');
      } else {
        // Handle the case where the API returns a different format
        // Try to extract the parts from a string response if needed
        const rawText = typeof data === 'string' ? data : JSON.stringify(data);
        setTransformedResponse(rawText);
        message.info('Transformed request received in unexpected format');
      }
    } catch (err) {
      console.error('Error transforming request:', err);
      message.error('Failed to transform request');
    } finally {
      setIsLoading(false);
    }
  };

  // Add this handler function near your other handlers
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault(); // Prevent default behavior
      handleTransform();
    }
  };

  return (
    <div className={styles['transform-playground']}>
      <div className={styles['playground-container']}>
        {/* Original Request Panel */}
        <div className={styles.panel}>
          <div className={styles['panel-header']}>
            <h2>Original Request</h2>
            <p>The request you would send to LiteLLM /chat/completions endpoint.</p>
          </div>
          
          <textarea 
            className={styles['code-input']}
            value={originalRequestJSON}
            onChange={(e) => setOriginalRequestJSON(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Press Cmd/Ctrl + Enter to transform"
          />
          
          <div className={styles['panel-footer']}>
            <Button 
              type="primary"
              onClick={handleTransform}
              loading={isLoading}
              className={styles['transform-button']}
            >
              <span>Transform</span>
              <span>→</span>
            </Button>
          </div>
        </div>
        
        {/* Transformed Request Panel */}
        <div className={styles.panel}>
          <div className={styles['panel-header']}>
            <h2>Transformed Request</h2>
            <p>How LiteLLM transforms your request for the specified provider.</p>
            <p className={styles.note}>Note: Sensitive headers are not shown.</p>
          </div>
          
          <div className={styles['code-output-container']}>
            <pre className={styles['code-output']}>
              {transformedResponse || `curl -X POST \\
  https://api.openai.com/v1/chat/completions \\
  -H 'Authorization: Bearer sk-xxx' \\
  -H 'Content-Type: application/json' \\
  -d '{
  "model": "gpt-4",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    }
  ],
  "temperature": 0.7
  }'`}
            </pre>
            
            <Button 
              type="text"
              icon={<CopyOutlined />}
              className={styles['copy-button']}
              onClick={() => {
                navigator.clipboard.writeText(transformedResponse || '');
                message.success('Copied to clipboard');
              }}
            />
          </div>
        </div>
      </div>
      <div className={styles.footer}>
        <p>Found an error? File an issue <a href="https://github.com/BerriAI/litellm/issues" target="_blank" rel="noopener noreferrer">here</a>.</p>
      </div>
    </div>
  );
};

export default TransformRequestPanel; 