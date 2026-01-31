/**
 * OutputCard - Displays output message with token count and cost
 * Datadog-style: header with icon/metrics, content below
 */

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
  const handleCopy = () => {
    if (!message) return;
    
    const content = JSON.stringify(message, null, 2);
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
        />
        <div style={{ padding: '12px 14px' }}>
          <Text type="secondary" style={{ fontSize: 13, fontStyle: 'italic' }}>
            No response data available
          </Text>
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
      />

      {/* Content */}
      <div style={{ padding: '12px 14px' }}>
        <SimpleMessageBlock
          label="ASSISTANT"
          content={message.content}
          toolCalls={message.toolCalls}
        />
      </div>
    </div>
  );
}

