import React from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ArrowUp } from "lucide-react";

interface MessageInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  disabled?: boolean;
  hasAttachment?: boolean;
  uploadComponent?: React.ReactNode;
}

export function MessageInput({
  value,
  onChange,
  onSend,
  disabled,
  hasAttachment,
  uploadComponent,
}: MessageInputProps) {
  const canSend =
    !disabled && (value.trim().length > 0 || Boolean(hasAttachment));

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
      <div className="flex items-center flex-1 bg-background border border-border rounded-xl px-3 py-1 min-h-[44px]">
        {uploadComponent && (
          <div className="flex-shrink-0 mr-2">{uploadComponent}</div>
        )}
        <Textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message... (Shift+Enter for new line)"
          disabled={disabled}
          rows={1}
          className="flex-1 resize-none border-0 shadow-none bg-transparent p-0 py-1 text-sm leading-5 min-h-[20px]"
        />
        <Button
          onClick={onSend}
          disabled={!canSend}
          size="icon"
          className="rounded-full h-8 w-8"
        >
          <ArrowUp className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
