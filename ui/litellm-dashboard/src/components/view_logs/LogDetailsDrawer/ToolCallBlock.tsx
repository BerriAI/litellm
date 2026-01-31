/**
 * ToolCallBlock - Displays tool call with white background
 * Minimal, monochrome styling
 */

import { useState } from 'react';
import { Typography, Button, message } from 'antd';
import { CopyOutlined } from '@ant-design/icons';
import { ToolCall } from './prettyMessagesTypes';

const { Text } = Typography;

interface ToolCallBlockProps {
  tool: ToolCall;
}

export function ToolCallBlock({ tool }: ToolCallBlockProps) {
  const [isHovered, setIsHovered] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(tool.arguments, null, 2));
    message.success('Tool arguments copied');
  };

  return (
    <div
      style={{
        background: '#fff',
        border: '1px solid #e8e8e8',
        borderRadius: 4,
        padding: '8px 12px',
        marginTop: 8,
        fontFamily: 'monospace',
        fontSize: 12,
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Tool Name Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: Object.keys(tool.arguments).length > 0 ? 6 : 0,
        }}
      >
        <Text strong style={{ fontSize: 12, color: '#262626' }}>
          {tool.name}
        </Text>
        <Button
          type="text"
          size="small"
          icon={<CopyOutlined />}
          style={{
            opacity: isHovered ? 1 : 0,
            transition: 'opacity 0.2s',
          }}
          onClick={handleCopy}
        />
      </div>

      {/* Tool Arguments */}
      {Object.keys(tool.arguments).length > 0 && (
        <div>
          {Object.entries(tool.arguments).map(([key, value]) => (
            <div key={key} style={{ marginBottom: 2 }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {key}:{' '}
              </Text>
              <Text style={{ fontSize: 12 }}>{JSON.stringify(value)}</Text>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
