/**
 * PrettyMessagesView - Chat-style view of request and response messages
 * Replaces raw JSON with scannable, readable message cards
 */

import { Typography } from 'antd';
import { parseMessages } from './prettyMessagesUtils';
import { MessageCard } from './MessageCard';
import { HistorySection } from './HistorySection';

const { Text } = Typography;

interface PrettyMessagesViewProps {
  request: any;
  response: any;
}

export function PrettyMessagesView({ request, response }: PrettyMessagesViewProps) {
  const { requestMessages, responseMessage } = parseMessages(request, response);

  // Separate system, history, and last user message
  const systemMessage = requestMessages.find((m) => m.role === 'system');
  const nonSystemMessages = requestMessages.filter((m) => m.role !== 'system');
  const lastUserMessage =
    nonSystemMessages.length > 0 ? nonSystemMessages[nonSystemMessages.length - 1] : null;
  const historyMessages = nonSystemMessages.slice(0, -1);

  return (
    <div style={{ paddingTop: 4, paddingBottom: 16 }}>
      {/* REQUEST SECTION */}
      <div>
        <Text
          type="secondary"
          style={{
            fontSize: 12,
            marginBottom: 16,
            display: 'block',
          }}
        >
          Request ({requestMessages.length} message{requestMessages.length !== 1 ? 's' : ''})
        </Text>

        {/* System Message - Collapsed by default */}
        {systemMessage && <MessageCard message={systemMessage} defaultCollapsed={true} />}

        {/* History - Collapsed if > 0 messages */}
        {historyMessages.length > 0 && <HistorySection messages={historyMessages} />}

        {/* Last User Message - Always expanded */}
        {lastUserMessage && lastUserMessage.role === 'user' && (
          <MessageCard message={lastUserMessage} defaultCollapsed={false} />
        )}

        {/* Fallback if no messages */}
        {requestMessages.length === 0 && (
          <div
            style={{
              textAlign: 'center',
              padding: 20,
              color: '#8c8c8c',
              fontStyle: 'italic',
              fontSize: 13,
            }}
          >
            No request messages available
          </div>
        )}
      </div>

      {/* Section Divider */}
      <div
        style={{
          borderTop: '1px solid #f0f0f0',
          margin: '20px 0',
          paddingTop: 16,
        }}
      >
        <Text
          type="secondary"
          style={{
            fontSize: 12,
            marginBottom: 16,
            display: 'block',
          }}
        >
          Response
        </Text>

        {/* RESPONSE SECTION */}
        {responseMessage ? (
          <MessageCard message={responseMessage} defaultCollapsed={false} showToolCalls />
        ) : (
          <div
            style={{
              textAlign: 'center',
              padding: 20,
              color: '#8c8c8c',
              fontStyle: 'italic',
              fontSize: 13,
            }}
          >
            Response data not available
          </div>
        )}
      </div>
    </div>
  );
}
