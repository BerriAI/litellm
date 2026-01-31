/**
 * HistorySection - Collapsible section for displaying message history
 * Shows a summary when collapsed, full messages when expanded
 */

import { useState } from 'react';
import { Typography } from 'antd';
import { UpOutlined, DownOutlined } from '@ant-design/icons';
import { ParsedMessage } from './prettyMessagesTypes';
import { MessageCard } from './MessageCard';
import { ROLE_STYLES } from './prettyMessagesUtils';

const { Text } = Typography;

interface HistorySectionProps {
  messages: ParsedMessage[];
}

export function HistorySection({ messages }: HistorySectionProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (messages.length === 0) return null;

  // Build summary strip showing message flow
  const summary = messages
    .map((m) => {
      const style = ROLE_STYLES[m.role] || ROLE_STYLES.user;
      return style.label;
    })
    .join(' → ');

  return (
    <div style={{ margin: '16px 0' }}>
      {/* Collapsed: Dashed line with label */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        style={{
          display: 'flex',
          alignItems: 'center',
          cursor: 'pointer',
          gap: 12,
        }}
      >
        <div style={{ flex: 1, borderTop: '1px dashed #d9d9d9' }} />
        <Text type="secondary" style={{ fontSize: 12 }}>
          History ({messages.length} message{messages.length !== 1 ? 's' : ''})
          {!isExpanded && ` · ${summary}`}
        </Text>
        {isExpanded ? (
          <UpOutlined style={{ color: '#8c8c8c', fontSize: 10 }} />
        ) : (
          <DownOutlined style={{ color: '#8c8c8c', fontSize: 10 }} />
        )}
        <div style={{ flex: 1, borderTop: '1px dashed #d9d9d9' }} />
      </div>

      {/* Expanded View - Full Messages with subtle indent */}
      {isExpanded && (
        <div
          style={{
            marginTop: 16,
            paddingLeft: 16,
            borderLeft: '1px solid #f0f0f0',
          }}
        >
          {messages.map((msg, index) => (
            <MessageCard
              key={index}
              message={msg}
              defaultCollapsed={msg.content?.length > 500}
            />
          ))}
        </div>
      )}
    </div>
  );
}
