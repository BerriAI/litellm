/**
 * MessageBlock - Displays a single message with role label
 * No colors, minimal gray styling
 */

import { useState } from 'react';
import { Typography, Button } from 'antd';
import { ToolCall } from './prettyMessagesTypes';
import { ToolCallBlock } from './ToolCallBlock';

const { Text } = Typography;

interface MessageBlockProps {
  role: string;
  content?: string;
  toolCalls?: ToolCall[];
}

const TRUNCATE_LENGTH = 500;

export function MessageBlock({ role, content, toolCalls }: MessageBlockProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const hasContent = content && content.length > 0;
  const hasToolCalls = toolCalls && toolCalls.length > 0;
  const isLong = hasContent && content.length > TRUNCATE_LENGTH;
  const shouldTruncate = isLong && !isExpanded;

  // If no content and no tool calls, don't render anything
  if (!hasContent && !hasToolCalls) {
    return null;
  }

  return (
    <div style={{ marginBottom: 12 }}>
      {/* Role Label */}
      <Text
        type="secondary"
        style={{
          fontSize: 11,
          display: 'block',
          marginBottom: 4,
          color: '#8c8c8c',
        }}
      >
        {role}
      </Text>

      {/* Content */}
      {hasContent && (
        <div
          style={{
            fontSize: 13,
            lineHeight: 1.6,
            color: '#262626',
            marginBottom: hasToolCalls ? 8 : 0,
          }}
        >
          {shouldTruncate ? (
            <>
              {content.slice(0, TRUNCATE_LENGTH)}...
              <Button
                type="link"
                size="small"
                onClick={() => setIsExpanded(true)}
                style={{ padding: '0 4px', fontSize: 12 }}
              >
                Show more
              </Button>
            </>
          ) : (
            <>
              {content}
              {isLong && isExpanded && (
                <Button
                  type="link"
                  size="small"
                  onClick={() => setIsExpanded(false)}
                  style={{
                    padding: '0 4px',
                    fontSize: 12,
                    display: 'block',
                    marginTop: 4,
                  }}
                >
                  Show less
                </Button>
              )}
            </>
          )}
        </div>
      )}

      {/* Tool Calls */}
      {hasToolCalls && (
        <div>
          {toolCalls.map((tool, index) => (
            <ToolCallBlock key={tool.id || index} tool={tool} />
          ))}
        </div>
      )}
    </div>
  );
}
