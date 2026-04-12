/**
 * InputCard - Displays all input messages with token count and cost
 * Datadog-style: header with icon/metrics, content below
 */

import { useState } from 'react';
import { message } from 'antd';
import { ParsedMessage } from './prettyMessagesTypes';
import { SectionHeader } from './SectionHeader';
import { CollapsibleMessage } from './CollapsibleMessage';
import { HistoryTree } from './HistoryTree';
import { SimpleMessageBlock } from './SimpleMessageBlock';

interface InputCardProps {
  messages: ParsedMessage[];
  promptTokens?: number;
  inputCost?: number;
}

export function InputCard({ messages, promptTokens, inputCost }: InputCardProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  if (messages.length === 0) {
    return null;
  }

  // Separate system, history, and last message
  const systemMessage = messages.find((m) => m.role === 'system');
  const nonSystemMessages = messages.filter((m) => m.role !== 'system');
  const lastMessage = nonSystemMessages.length > 0 ? nonSystemMessages[nonSystemMessages.length - 1] : null;
  const historyMessages = nonSystemMessages.slice(0, -1);

  const handleCopy = () => {
    const content = lastMessage?.content || '';
    navigator.clipboard.writeText(content);
    message.success('Input copied');
  };

  return (
    <div
      style={{
        border: '1px solid #f0f0f0',
        borderRadius: 6,
        marginBottom: 8,
        overflow: 'hidden',
      }}
    >
      {/* Datadog-style Header */}
      <SectionHeader
        type="input"
        tokens={promptTokens}
        cost={inputCost}
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
          {/* System Message - Collapsible with arrow */}
          {systemMessage && (
            <CollapsibleMessage
              label="SYSTEM"
              content={systemMessage.content}
              defaultExpanded={!!(systemMessage.content && systemMessage.content.length < 200)}
            />
          )}

          {/* History - Tree style, collapsed by default */}
          {historyMessages.length > 0 && <HistoryTree messages={historyMessages} />}

          {/* Last User Message - Always visible */}
          {lastMessage && (
            <SimpleMessageBlock
              label={lastMessage.role.toUpperCase()}
              content={lastMessage.content}
              toolCalls={lastMessage.toolCalls}
            />
          )}
        </div>
      </div>
    </div>
  );
}
