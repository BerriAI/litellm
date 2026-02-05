/**
 * SimpleToolCallBlock - Simple tool call display without copy button
 * Used in compact/tree views
 */

import { Typography } from 'antd';
import { ToolCall } from './prettyMessagesTypes';

const { Text } = Typography;

interface SimpleToolCallBlockProps {
  tool: ToolCall;
  compact?: boolean;
}

export function SimpleToolCallBlock({ tool, compact = false }: SimpleToolCallBlockProps) {
  return (
    <div
      style={{
        background: '#f8f9fa',
        border: '1px solid #e9ecef',
        borderRadius: 6,
        padding: compact ? '6px 10px' : '10px 14px',
        marginTop: 8,
        fontFamily: 'monospace',
        fontSize: 12,
        position: 'relative',
      }}
    >
      {/* Function badge */}
      <div
        style={{
          position: 'absolute',
          top: -8,
          left: 12,
          background: '#fff',
          padding: '0 6px',
          fontSize: 10,
          color: '#8c8c8c',
          border: '1px solid #e9ecef',
          borderRadius: 3,
        }}
      >
        function
      </div>

      <Text strong style={{ fontSize: 13, display: 'block', marginBottom: 6 }}>
        {tool.name}
      </Text>

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
