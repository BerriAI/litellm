import React from "react";
import { ArrowUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface MessageInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  disabled?: boolean;
  hasAttachment?: boolean;
  uploadComponent?: React.ReactNode;
}

export function MessageInput({ value, onChange, onSend, disabled, hasAttachment, uploadComponent }: MessageInputProps) {
  const canSend = !disabled && (value.trim().length > 0 || Boolean(hasAttachment));

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canSend) {
        onSend();
      }
    }
  };

  return (
    <div className="flex items-center gap-2">
      <div className="flex items-center flex-1 bg-card border border-border rounded-xl px-3 py-1 min-h-[44px]">
        {uploadComponent && <div className="shrink-0 mr-2">{uploadComponent}</div>}
        <Textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message... (Shift+Enter for new line)"
          disabled={disabled}
          rows={1}
          className="max-h-20 min-h-0 flex-1 resize-none overflow-y-auto border-0 bg-transparent px-0 py-1 text-sm leading-5 shadow-none focus-visible:ring-0"
        />
        <Button
          onClick={onSend}
          disabled={!canSend}
          size="icon-sm"
          variant="outline"
          className="rounded-full"
          aria-label="Send message"
        >
          <ArrowUp />
        </Button>
      </div>
    </div>
  );
}
