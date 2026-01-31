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
  const [isHovered, setIsHovered] = useState(false);
  const charCount = content?.length || 0;

  if (!content || charCount === 0) {
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
          {label}
        </Text>
        <Text type="secondary" style={{ fontSize: 10 }}>
          ({charCount.toLocaleString()} chars)
        </Text>
      </div>

      {/* Content with smooth animation */}
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
            fontSize: 13,
            lineHeight: 1.7,
            color: '#262626',
            borderLeft: '1px solid #f0f0f0',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {content}
        </div>
      </div>
    </div>
  );
}
