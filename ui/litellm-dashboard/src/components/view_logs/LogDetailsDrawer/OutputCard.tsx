/**
 * OutputCard - Displays output message with token count and cost
 * Datadog-style: header with icon/metrics, content below
 */

import { useState } from 'react';
import { Typography, message as antdMessage } from 'antd';
import { ParsedMessage } from './prettyMessagesTypes';
import { SectionHeader } from './SectionHeader';
import { SimpleMessageBlock } from './SimpleMessageBlock';

const { Text } = Typography;

interface OutputCardProps {
  message: ParsedMessage | null;
  completionTokens?: number;
  outputCost?: number;
}

export function OutputCard({ message, completionTokens, outputCost }: OutputCardProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const handleCopy = () => {
    if (!message) return;
    
    const content = message.content || '';
    navigator.clipboard.writeText(content);
    antdMessage.success('Output copied');
  };

  if (!message) {
    return (
      <div
        style={{
          border: '1px solid #f0f0f0',
          borderRadius: 6,
          overflow: 'hidden',
        }}
      >
        <SectionHeader
          type="output"
          tokens={completionTokens}
          cost={outputCost}
          onCopy={handleCopy}
          isCollapsed={isCollapsed}
          onToggleCollapse={() => setIsCollapsed(!isCollapsed)}
        />
        <div
          style={{
            maxHeight: isCollapsed ? '0px' : '10000px',
            overflow: 'hidden',
            transition: 'max-height 0.3s ease-out, opacity 0.3s ease-out',
            opacity: isCollapsed ? 0 : 1,
          }}
        >
          <div style={{ padding: '12px 16px' }}>
            <Text type="secondary" style={{ fontSize: 13, fontStyle: 'italic' }}>
              No response data available
            </Text>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        border: '1px solid #f0f0f0',
        borderRadius: 6,
        overflow: 'hidden',
      }}
    >
      {/* Datadog-style Header */}
      <SectionHeader
        type="output"
        tokens={completionTokens}
        cost={outputCost}
        onCopy={handleCopy}
        isCollapsed={isCollapsed}
        onToggleCollapse={() => setIsCollapsed(!isCollapsed)}
      />

      {/* Content */}
      <div
        style={{
          maxHeight: isCollapsed ? '0px' : '10000px',
          overflow: 'hidden',
          transition: 'max-height 0.3s ease-out, opacity 0.3s ease-out',
          opacity: isCollapsed ? 0 : 1,
        }}
      >
        <div style={{ padding: '12px 16px' }}>
          <SimpleMessageBlock
            label="ASSISTANT"
            content={message.content}
            toolCalls={message.toolCalls}
          />
        </div>
      </div>
    </div>
  );
}

