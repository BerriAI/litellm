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
  const [isHovered, setIsHovered] = useState(false);

  if (messages.length === 0) {
    return null;
  }

  return (
    <div style={{ marginBottom: 8 }}>
      {/* Clickable Header with hover state */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          cursor: 'pointer',
          padding: '4px 0',
          borderRadius: 4,
          background: isHovered ? '#f5f5f5' : 'transparent',
          transition: 'background 0.15s ease',
          marginBottom: isExpanded ? 4 : 0,
        }}
      >
        {isExpanded ? (
          <DownOutlined style={{ fontSize: 10, color: '#8c8c8c' }} />
        ) : (
          <RightOutlined style={{ fontSize: 10, color: '#8c8c8c' }} />
        )}
        <Text type="secondary" style={{ fontSize: 10, letterSpacing: '0.5px', textTransform: 'uppercase' }}>
          HISTORY ({messages.length} message{messages.length !== 1 ? 's' : ''})
        </Text>
      </div>

      {/* Expanded Tree Content with smooth animation */}
      <div
        style={{
          maxHeight: isExpanded ? '2000px' : '0px',
          overflow: 'hidden',
          transition: 'max-height 0.2s ease-out, opacity 0.2s ease-out',
          opacity: isExpanded ? 1 : 0,
        }}
      >
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
      </div>
    </div>
  );
}
