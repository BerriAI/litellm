import type { ClipboardEventHandler, ReactNode } from "react";
import { Send, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import "./moyai-chat.css";

interface MoyaiChatComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onCancel: () => void;
  placeholder: string;
  isLoading: boolean;
  submitDisabled: boolean;
  submitLabel: string;
  onPaste?: ClipboardEventHandler<HTMLTextAreaElement>;
  attachmentPreview?: ReactNode;
  tools?: ReactNode;
  modelPicker?: ReactNode;
  inConversation?: boolean;
}

export default function MoyaiChatComposer({
  value,
  onChange,
  onSubmit,
  onCancel,
  placeholder,
  isLoading,
  submitDisabled,
  submitLabel,
  onPaste,
  attachmentPreview,
  tools,
  modelPicker,
  inConversation = false,
}: MoyaiChatComposerProps) {
  const send = () => {
    if (!submitDisabled && !isLoading) onSubmit();
  };
  const sendButton = isLoading ? (
    <Button type="button" size="icon" aria-label="Stop request" onClick={onCancel}>
      <Square className="size-4" />
    </Button>
  ) : (
    <Button type="button" size="icon" aria-label={submitLabel} disabled={submitDisabled} onClick={send}>
      <Send className="size-4" />
    </Button>
  );
  return (
    <div className={inConversation ? "moyai-chat-conversation-composer" : "moyai-chat-home-composer"}>
      <div className="moyai-chat-composer-entry">
        {attachmentPreview}
        <div className="moyai-chat-home-editor-input">
          <Textarea
            value={value}
            onChange={(event) => onChange(event.target.value)}
            placeholder={placeholder}
            aria-label="Message"
            onPaste={onPaste}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault();
                send();
              }
            }}
            className="moyai-chat-editor"
          />
        </div>
        {inConversation && <div className="moyai-chat-entry-send">{sendButton}</div>}
      </div>
      <div className="moyai-chat-composer-actions">
        <div className="moyai-chat-home-composer-tools">{tools}</div>
        <div className="moyai-chat-send-actions">
          {modelPicker}
          {!inConversation && sendButton}
        </div>
      </div>
    </div>
  );
}
