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
        background: '#fafafa',
        border: '1px solid #f0f0f0',
        borderRadius: 4,
        padding: compact ? '6px 10px' : '8px 12px',
        marginTop: 8,
        fontFamily: 'monospace',
        fontSize: 12,
      }}
    >
      <Text strong style={{ fontSize: 12, display: 'block', marginBottom: 4 }}>
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
