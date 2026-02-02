/**
 * ToolCallCard - Display tool call information inline in assistant messages
 */

import { useState } from 'react';
import { Button, Typography, message } from 'antd';
import { CopyOutlined } from '@ant-design/icons';
import { ToolCall } from './prettyMessagesTypes';

const { Text } = Typography;

interface ToolCallCardProps {
  tool: ToolCall;
}

export function ToolCallCard({ tool }: ToolCallCardProps) {
  const [isHovered, setIsHovered] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(JSON.stringify(tool.arguments, null, 2));
    message.success('Tool arguments copied');
  };

  return (
    <div
      style={{
        background: '#fafafa',
        border: '1px solid #f0f0f0',
        borderRadius: 4,
        padding: '8px 12px',
        marginBottom: 8,
        fontFamily: 'monospace',
        fontSize: 12,
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Tool Header */}
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

      {/* Tool Arguments - Simple key: value format */}
      {Object.keys(tool.arguments).length > 0 && (
        <div>
          {Object.entries(tool.arguments).map(([key, value]) => (
            <div key={key} style={{ marginBottom: 2 }}>
              <Text type="secondary" style={{ fontSize: 12 }}>
                {key}:
              </Text>{' '}
              <Text code style={{ background: 'transparent', fontSize: 12 }}>
                {JSON.stringify(value)}
              </Text>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
