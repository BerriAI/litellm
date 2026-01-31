/**
 * CollapsibleMessage - Collapsible message with arrow and char count
 * Used for system messages
 */

import { useState } from 'react';
import { Typography } from 'antd';
import { DownOutlined, RightOutlined } from '@ant-design/icons';

const { Text } = Typography;

interface CollapsibleMessageProps {
  label: string;
  content?: string;
  defaultExpanded?: boolean;
}

export function CollapsibleMessage({ 
  label, 
  content, 
  defaultExpanded = false 
}: CollapsibleMessageProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const charCount = content?.length || 0;

  if (!content || charCount === 0) {
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
          marginBottom: isExpanded ? 6 : 0,
        }}
      >
        {isExpanded ? (
          <DownOutlined style={{ fontSize: 10, color: '#8c8c8c' }} />
        ) : (
          <RightOutlined style={{ fontSize: 10, color: '#8c8c8c' }} />
        )}
        <Text type="secondary" style={{ fontSize: 11 }}>
          {label}
        </Text>
        <Text type="secondary" style={{ fontSize: 11 }}>
          ({charCount.toLocaleString()} chars)
        </Text>
      </div>

      {/* Content */}
      {isExpanded && (
        <div
          style={{
            paddingLeft: 16,
            fontSize: 13,
            lineHeight: 1.6,
            color: '#262626',
            borderLeft: '1px solid #f0f0f0',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {content}
        </div>
      )}
    </div>
  );
}
