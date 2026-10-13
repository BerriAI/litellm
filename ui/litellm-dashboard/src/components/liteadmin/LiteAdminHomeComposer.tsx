import { useRef, type ReactNode } from "react";
import { ArrowUp, Box, CreditCard, KeyRound, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { InputGroup, InputGroupTextarea } from "@/components/ui/input-group";

const suggestions = [
  { label: "Create a key", icon: KeyRound },
  { label: "Add a model", icon: Box },
  { label: "Set a budget", icon: CreditCard },
];

export function LiteAdminHomeComposer({
  value,
  onChange,
  onSubmit,
  onCancel,
  disabled,
  isLoading,
  submitDisabled,
  modelPicker,
}: {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onCancel?: () => void;
  disabled: boolean;
  isLoading: boolean;
  submitDisabled: boolean;
  modelPicker: ReactNode;
}) {
  const inputRef = useRef<HTMLTextAreaElement>(null);
  return (
    <InputGroup className="h-auto flex-col rounded-xl border-border bg-card shadow-xs">
      <InputGroupTextarea
        ref={inputRef}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault();
            if (!submitDisabled && !isLoading) onSubmit();
          }
        }}
        placeholder="What would you like to do?"
        aria-label="Message LiteAdmin"
        disabled={disabled}
        rows={2}
        className="min-h-24 max-h-48 w-full rounded-t-xl px-5 pt-5 pb-2 text-sm placeholder:text-muted-foreground [field-sizing:content]"
      />
      <div className="flex w-full flex-wrap items-center justify-between gap-3 px-4 pb-4">
        <div className="flex flex-wrap gap-2">
          {suggestions.map(({ label, icon: Icon }) => (
            <Button
              key={label}
              type="button"
              variant="outline"
              size="sm"
              disabled={disabled}
              className="gap-2 bg-muted/30 font-normal text-muted-foreground shadow-none"
              onClick={() => {
                onChange(label);
                inputRef.current?.focus();
              }}
            >
              <Icon className="size-3.5" />
              {label}
            </Button>
          ))}
        </div>
        <div className="ml-auto flex min-w-0 max-w-full items-center gap-2">
          <div className="w-48 min-w-0 text-muted-foreground">{modelPicker}</div>
          {isLoading && onCancel ? (
            <Button type="button" size="icon-sm" aria-label="Stop request" onClick={onCancel}>
              <Square className="size-3.5 fill-current" />
            </Button>
          ) : (
            <Button
              type="button"
              size="icon-sm"
              aria-label="Send message"
              disabled={submitDisabled || isLoading}
              onClick={onSubmit}
              className="rounded-lg bg-info text-info-foreground shadow-xs hover:bg-info/90"
            >
              <ArrowUp className="size-4" />
            </Button>
          )}
        </div>
      </div>
    </InputGroup>
  );
}
