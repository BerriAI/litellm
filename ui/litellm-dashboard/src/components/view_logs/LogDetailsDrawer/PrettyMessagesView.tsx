/**
 * PrettyMessagesView - Datadog-style view with Input/Output cards
 * Two main cards showing request and response with token counts and costs
 */

import { parseMessages } from './prettyMessagesUtils';
import { InputCard } from './InputCard';
import { OutputCard } from './OutputCard';

interface PrettyMessagesViewProps {
  request: any;
  response: any;
  metrics?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    input_cost?: number;
    output_cost?: number;
  };
}

export function PrettyMessagesView({ request, response, metrics }: PrettyMessagesViewProps) {
  const { requestMessages, responseMessage } = parseMessages(request, response);

  return (
    <div>
      {/* Input Card */}
      <InputCard
        messages={requestMessages}
        promptTokens={metrics?.prompt_tokens}
        inputCost={metrics?.input_cost}
      />

      {/* Output Card */}
      <OutputCard
        message={responseMessage}
        completionTokens={metrics?.completion_tokens}
        outputCost={metrics?.output_cost}
      />
    </div>
  );
}
