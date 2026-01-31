/**
 * HistoryTree - Collapsible tree view for message history
 * Shows arrow indicator and message count
 */

import { useState } from 'react';
import { Typography } from 'antd';
import { DownOutlined, RightOutlined } from '@ant-design/icons';
import { ParsedMessage } from './prettyMessagesTypes';
import { SimpleMessageBlock } from './SimpleMessageBlock';

const { Text } = Typography;

interface HistoryTreeProps {
  messages: ParsedMessage[];
}

export function HistoryTree({ messages }: HistoryTreeProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (messages.length === 0) {
    return null;
  }

  return (
    <div style={{ marginBottom: 12 }}>
      {/* Clickable Header */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          cursor: 'pointer',
          marginBottom: isExpanded ? 8 : 0,
        }}
      >
        {isExpanded ? (
          <DownOutlined style={{ fontSize: 10, color: '#8c8c8c' }} />
        ) : (
          <RightOutlined style={{ fontSize: 10, color: '#8c8c8c' }} />
        )}
        <Text type="secondary" style={{ fontSize: 11 }}>
          HISTORY ({messages.length} message{messages.length !== 1 ? 's' : ''})
        </Text>
      </div>

      {/* Expanded Tree Content */}
      {isExpanded && (
        <div
          style={{
            paddingLeft: 16,
            borderLeft: '1px solid #f0f0f0',
          }}
        >
          {messages.map((msg, index) => (
            <SimpleMessageBlock
              key={index}
              label={msg.role.toUpperCase()}
              content={msg.content}
              toolCalls={msg.toolCalls}
              isCompact={true}
            />
          ))}
        </div>
      )}
    </div>
  );
}
