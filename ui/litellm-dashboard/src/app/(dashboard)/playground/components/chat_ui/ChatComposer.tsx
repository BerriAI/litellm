import React from "react";
import { ArrowUp, Code2, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupTextarea } from "@/components/ui/input-group";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/cva.config";

interface ChatComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onCancel?: () => void;
  placeholder: string;
  disabled?: boolean;
  isLoading?: boolean;
  submitDisabled?: boolean;
  tools?: React.ReactNode;
  body?: React.ReactNode;
  suggestions?: string[];
  showSuggestions?: boolean;
  onSuggestionSelect?: (suggestion: string) => void;
  className?: string;
}

export function ChatComposer({
  value,
  onChange,
  onSubmit,
  onCancel,
  placeholder,
  disabled = false,
  isLoading = false,
  submitDisabled = false,
  tools,
  body,
  suggestions = [],
  showSuggestions = false,
  onSuggestionSelect,
  className,
}: ChatComposerProps) {
  const submitIfAllowed = () => {
    if (!submitDisabled && !isLoading) {
      onSubmit();
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submitIfAllowed();
    }
  };

  return (
    <div className={cn("relative flex w-full flex-col gap-3", className)}>
      {showSuggestions && suggestions.length > 0 && (
        <div className="flex w-full flex-col gap-1.5" data-testid="chat-suggested-actions">
          {suggestions.map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              className="w-full truncate rounded-lg border border-border/50 bg-card/30 px-3 py-1.5 text-left text-[12px] leading-snug text-muted-foreground transition-colors hover:bg-card/60 hover:text-foreground"
              onClick={() => onSuggestionSelect?.(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}

      <div className="w-full">
        <InputGroup
          className={cn(
            "h-auto min-h-[7.5rem] flex-col overflow-hidden rounded-2xl border border-border bg-card",
            "shadow-[0_1px_2px_rgba(0,0,0,0.06),0_8px_24px_rgba(0,0,0,0.08)] ring-1 ring-black/5",
            "transition-[box-shadow,border-color,ring] duration-200",
            "has-[[data-slot=input-group-control]:focus-visible]:border-ring",
            "has-[[data-slot=input-group-control]:focus-visible]:shadow-[0_2px_8px_rgba(0,0,0,0.08),0_12px_32px_rgba(0,0,0,0.12)]",
            "has-[[data-slot=input-group-control]:focus-visible]:ring-2 has-[[data-slot=input-group-control]:focus-visible]:ring-ring/40",
          )}
        >
          {body ? (
            <div className="max-h-48 min-h-24 w-full overflow-y-auto px-3 pt-3">{body}</div>
          ) : (
            <InputGroupTextarea
              data-testid="chat-composer-input"
              value={value}
              disabled={disabled}
              placeholder={placeholder}
              rows={1}
              className="min-h-24 max-h-48 resize-none overflow-y-auto border-0 bg-transparent px-4 pt-3.5 pb-1.5 text-[13px] leading-relaxed shadow-none placeholder:text-muted-foreground/50 focus-visible:ring-0 [field-sizing:content]"
              onChange={(event) => onChange(event.target.value)}
              onKeyDown={handleKeyDown}
            />
          )}

          <InputGroupAddon
            align="block-end"
            className="justify-between gap-2 px-3 pb-3 pt-1"
            onClick={(event) => {
              if ((event.target as HTMLElement).closest("button")) {
                return;
              }
              event.currentTarget.parentElement?.querySelector<HTMLElement>("[data-slot=input-group-control]")?.focus();
            }}
          >
            <div className="flex min-w-0 items-center gap-1">{tools}</div>

            {isLoading && onCancel ? (
              <InputGroupButton
                type="button"
                size="icon-sm"
                aria-label="Stop request"
                data-testid="chat-stop-button"
                className="size-8 rounded-xl bg-foreground text-background hover:bg-foreground/90"
                onClick={onCancel}
              >
                <Square className="size-3.5 fill-current" />
              </InputGroupButton>
            ) : (
              <InputGroupButton
                type="button"
                size="icon-sm"
                aria-label="Send message"
                data-testid="chat-send-button"
                disabled={submitDisabled || isLoading}
                onClick={submitIfAllowed}
                className={cn(
                  "size-8 rounded-xl transition-all duration-200",
                  !submitDisabled && !isLoading
                    ? "bg-foreground text-background hover:opacity-90 active:scale-95"
                    : "cursor-not-allowed bg-muted text-muted-foreground/40",
                )}
              >
                <ArrowUp className="size-4" />
              </InputGroupButton>
            )}
          </InputGroupAddon>
        </InputGroup>
      </div>
    </div>
  );
}

interface CodeInterpreterToggleProps {
  enabled: boolean;
  onToggle: () => void;
}

export function CodeInterpreterToggle({ enabled, onToggle }: CodeInterpreterToggleProps) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            className={cn(
              "size-8 rounded-lg border border-border/40",
              enabled
                ? "border-info/20 bg-info/10 text-info hover:bg-info/15"
                : "text-muted-foreground hover:text-foreground",
            )}
            aria-label={enabled ? "Code Interpreter enabled (click to disable)" : "Enable Code Interpreter"}
            onClick={onToggle}
          />
        }
      >
        <Code2 className="size-4" />
      </TooltipTrigger>
      <TooltipContent>
        {enabled ? "Code Interpreter enabled (click to disable)" : "Enable Code Interpreter"}
      </TooltipContent>
    </Tooltip>
  );
}

export default ChatComposer;
