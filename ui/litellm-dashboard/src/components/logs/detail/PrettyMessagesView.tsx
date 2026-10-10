/**
 * PrettyMessagesView - Datadog-style view with Input/Output cards
 * Two main cards showing request and response with token counts and costs.
 * Detects realtime API responses and renders a specialized view.
 */

import { parseMessages, parseResponsesWebSocketTurns } from "./prettyMessagesUtils";
import { ResponsesWebSocketPrettyView } from "./ResponsesWebSocketPrettyView";
import { InputCard } from "./InputCard";
import { OutputCard } from "./OutputCard";
import { isRealtimeResponse, RealtimePrettyView } from "./RealtimePrettyView";

interface PrettyMessagesViewProps {
  request: unknown;
  response: unknown;
  metrics?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    input_cost?: number;
    output_cost?: number;
  };
}

export function PrettyMessagesView({ request, response, metrics }: PrettyMessagesViewProps) {
  if (isRealtimeResponse(response)) {
    return <RealtimePrettyView response={response} metrics={metrics} />;
  }

  const { requestMessages, responseMessage } = parseMessages(request, response);
  const turns = parseResponsesWebSocketTurns(response);

  return (
    <div>
      {/* Input Card */}
      <InputCard messages={requestMessages} promptTokens={metrics?.prompt_tokens} inputCost={metrics?.input_cost} />

      {/* Output Card */}
      {turns !== null ? (
        <ResponsesWebSocketPrettyView
          turns={turns}
          completionTokens={metrics?.completion_tokens}
          outputCost={metrics?.output_cost}
        />
      ) : (
        <OutputCard
          message={responseMessage}
          completionTokens={metrics?.completion_tokens}
          outputCost={metrics?.output_cost}
        />
      )}
    </div>
  );
}
