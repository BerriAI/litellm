/**
 * MessageCard - Display individual message with role-based styling
 * Features: collapsible long content, copy button, tool calls display
 */

import { useState } from 'react';
import { Button, Typography, message as antdMessage } from 'antd';
import { CopyOutlined } from '@ant-design/icons';
import { ParsedMessage } from './prettyMessagesTypes';
import { ROLE_STYLES } from './prettyMessagesUtils';
import { ToolCallCard } from './ToolCallCard';

const { Text } = Typography;

interface MessageCardProps {
  message: ParsedMessage;
  defaultCollapsed?: boolean;
  showToolCalls?: boolean;
}

const TRUNCATE_LENGTH = 500;

export function MessageCard({
  message,
  defaultCollapsed = false,
  showToolCalls = false,
}: MessageCardProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);
  const [isHovered, setIsHovered] = useState(false);

  const style = ROLE_STYLES[message.role] || ROLE_STYLES.user;
  const content = message.content || '';
  const isLong = content.length > TRUNCATE_LENGTH;
  const shouldTruncate = isCollapsed && isLong;
  
  // Don't show empty content for assistant messages with tool calls
  const hasContent = content.length > 0;
  const hasToolCalls = showToolCalls && message.toolCalls && message.toolCalls.length > 0;
  
  // If assistant message with no content but has tool calls, skip null display
  if (message.role === 'assistant' && !hasContent && hasToolCalls) {
    return (
      <div
        style={{ marginBottom: 16 }}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        {/* Role Label Row */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 6,
          }}
        >
          <Text
            strong
            style={{
              fontSize: 11,
              color: style.labelColor,
              letterSpacing: '0.5px',
            }}
          >
            {style.label}
          </Text>
        </div>

        {/* Tool Calls with left border */}
        <div
          style={{
            borderLeft: `2px solid ${style.borderColor}`,
            paddingLeft: 12,
          }}
        >
          {message.toolCalls!.map((tool, index) => (
            <ToolCallCard key={tool.id || index} tool={tool} />
          ))}
        </div>
      </div>
    );
  }

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    antdMessage.success('Message copied');
  };

  return (
    <div
      style={{ marginBottom: 16 }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Role Label Row */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 6,
        }}
      >
        <Text
          strong
          style={{
            fontSize: 11,
            color: style.labelColor,
            letterSpacing: '0.5px',
          }}
        >
          {style.label}
          {isLong && (
            <Text type="secondary" style={{ marginLeft: 8, fontWeight: 'normal', fontSize: 11 }}>
              ({content.length.toLocaleString()} chars)
            </Text>
          )}
        </Text>

        {/* Copy Button - Show on hover */}
        {hasContent && (
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
        )}
      </div>

      {/* Content with left border accent */}
      {hasContent && (
        <div
          style={{
            borderLeft: `2px solid ${style.borderColor}`,
            paddingLeft: 12,
            fontSize: 13,
            lineHeight: 1.6,
            color: '#262626',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
          }}
        >
          {shouldTruncate ? (
            <>
              {content.slice(0, TRUNCATE_LENGTH)}...
              <Button
                type="link"
                size="small"
                onClick={() => setIsCollapsed(false)}
                style={{ padding: '0 4px', fontSize: 12 }}
              >
                Show more
              </Button>
            </>
          ) : (
            <>
              {content}
              {isLong && !isCollapsed && (
                <Button
                  type="link"
                  size="small"
                  onClick={() => setIsCollapsed(true)}
                  style={{
                    padding: '0 4px',
                    display: 'block',
                    marginTop: 4,
                    fontSize: 12,
                  }}
                >
                  Show less
                </Button>
              )}
            </>
          )}
        </div>
      )}

      {/* Tool Calls (for assistant messages) */}
      {hasToolCalls && hasContent && (
        <div
          style={{
            borderLeft: `2px solid ${style.borderColor}`,
            paddingLeft: 12,
            marginTop: 8,
          }}
        >
          {message.toolCalls!.map((tool, index) => (
            <ToolCallCard key={tool.id || index} tool={tool} />
          ))}
        </div>
      )}
    </div>
  );
}
