import React from "react";
import { ArrowUp } from "lucide-react";
import { useTranslation } from "react-i18next";
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
  const { t } = useTranslation();
  return (
    <div className="flex items-center gap-2">
      <div className="flex items-center flex-1 bg-background border border-border rounded-xl px-3 py-1 min-h-[44px]">
        <Textarea
          value={inputMessage}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={t("prompts.conversation.input.placeholder")}
          disabled={isLoading}
          rows={1}
          className="field-sizing-content max-h-24 min-h-8 flex-1 resize-none overflow-y-auto border-0 bg-transparent px-0 py-1 text-sm shadow-none focus-visible:ring-0"
        />

        <Button
          type="button"
          size="icon-sm"
          onClick={onSend}
          disabled={isDisabled}
          className="ml-2 shrink-0 rounded-full"
          aria-label={t("prompts.conversation.input.send")}
        >
          <ArrowUp aria-hidden="true" />
        </Button>
      </div>

      {isLoading && (
        <Button type="button" variant="destructive" onClick={onCancel}>
          {t("prompts.conversation.input.cancel")}
        </Button>
      )}
    </div>
  );
};

export default MessageInput;
