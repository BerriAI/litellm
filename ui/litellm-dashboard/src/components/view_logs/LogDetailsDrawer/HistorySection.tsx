/**
 * HistoryDivider - Collapsible divider for message history
 * Dashed line with expandable content
 */

import { useState } from 'react';
import { Typography } from 'antd';
import { UpOutlined, DownOutlined } from '@ant-design/icons';
import { ParsedMessage } from './prettyMessagesTypes';
import { MessageBlock } from './MessageBlock';

const { Text } = Typography;

interface HistoryDividerProps {
  messages: ParsedMessage[];
}

export function HistoryDivider({ messages }: HistoryDividerProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (messages.length === 0) return null;

  return (
    <div style={{ margin: '12px 0' }}>
      {/* Dashed Divider with Label */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        style={{
          display: 'flex',
          alignItems: 'center',
          cursor: 'pointer',
          gap: 8,
        }}
      >
        <div style={{ flex: 1, borderTop: '1px dashed #d9d9d9' }} />
        <Text type="secondary" style={{ fontSize: 11 }}>
          History ({messages.length})
        </Text>
        {isExpanded ? (
          <UpOutlined style={{ fontSize: 10, color: '#8c8c8c' }} />
        ) : (
          <DownOutlined style={{ fontSize: 10, color: '#8c8c8c' }} />
        )}
        <div style={{ flex: 1, borderTop: '1px dashed #d9d9d9' }} />
      </div>

      {/* Expanded View - Full Messages */}
      {isExpanded && (
        <div style={{ marginTop: 12 }}>
          {messages.map((msg, index) => (
            <MessageBlock
              key={index}
              role={msg.role.toUpperCase()}
              content={msg.content}
              toolCalls={msg.toolCalls}
            />
          ))}
        </div>
      )}
    </div>
  );
}
