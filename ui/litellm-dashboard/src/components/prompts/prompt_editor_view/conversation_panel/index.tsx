import React from "react";
import { Eraser } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ConversationPanelProps } from "./types";
import { useConversation } from "./useConversation";
import VariableInput from "./VariableInput";
import MessageList from "./MessageList";
import VariableWarning from "./VariableWarning";
import MessageInput from "./MessageInput";

const ConversationPanel: React.FC<ConversationPanelProps> = ({
  prompt,
  accessToken,
}) => {
  const {
    isLoading,
    messages,
    inputMessage,
    variables,
    variablesFilled,
    extractedVariables,
    allVariablesFilled,
    messagesEndRef,
    setInputMessage,
    handleSendMessage,
    handleCancelRequest,
    handleClearConversation,
    handleKeyDown,
    handleVariableChange,
  } = useConversation(prompt, accessToken);

  return (
    <div className="flex flex-col h-full bg-background">
      {!variablesFilled && (
        <VariableInput
          extractedVariables={extractedVariables}
          variables={variables}
          onVariableChange={handleVariableChange}
        />
      )}

      {messages.length > 0 && (
        <div className="p-3 border-b border-border bg-background flex justify-end">
          <Button
            variant="outline"
            size="sm"
            onClick={handleClearConversation}
          >
            <Eraser className="h-3.5 w-3.5" />
            Clear Chat
          </Button>
        </div>
      )}

      <MessageList
        messages={messages}
        isLoading={isLoading}
        hasVariables={extractedVariables.length > 0}
        messagesEndRef={messagesEndRef}
      />

      <div className="p-4 border-t border-border bg-background">
        <VariableWarning
          extractedVariables={extractedVariables}
          variables={variables}
        />

        <MessageInput
          inputMessage={inputMessage}
          isLoading={isLoading}
          isDisabled={
            isLoading ||
            !inputMessage.trim() ||
            (extractedVariables.length > 0 && !allVariablesFilled)
          }
          onInputChange={setInputMessage}
          onSend={handleSendMessage}
          onKeyDown={handleKeyDown}
          onCancel={handleCancelRequest}
        />
      </div>
    </div>
  );
};

export default ConversationPanel;
