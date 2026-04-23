import React from "react";
import { ArrowUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface MessageInputProps {
  inputMessage: string;
  isLoading: boolean;
  isDisabled: boolean;
  onInputChange: (value: string) => void;
  onSend: () => void;
  onKeyDown: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onCancel: () => void;
}

const MessageInput: React.FC<MessageInputProps> = ({
  inputMessage,
  isLoading,
  isDisabled,
  onInputChange,
  onSend,
  onKeyDown,
  onCancel,
}) => {
  return (
    <div className="flex items-center gap-2">
      <div className="flex items-center flex-1 bg-background border border-border rounded-xl px-3 py-1 min-h-[44px]">
        <Textarea
          value={inputMessage}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type your message... (Shift+Enter for new line)"
          disabled={isLoading}
          rows={1}
          className="flex-1 resize-none border-0 shadow-none bg-transparent p-0 py-1 text-sm leading-5 min-h-[20px]"
        />

        <Button
          onClick={onSend}
          disabled={isDisabled}
          size="icon"
          className="flex-shrink-0 ml-2 h-8 w-8 rounded-full"
        >
          <ArrowUp className="h-3.5 w-3.5" />
        </Button>
      </div>

      {isLoading && (
        <Button
          variant="outline"
          onClick={onCancel}
          className="bg-destructive/10 hover:bg-destructive/20 text-destructive border-destructive/20"
        >
          Cancel
        </Button>
      )}
    </div>
  );
};

export default MessageInput;
