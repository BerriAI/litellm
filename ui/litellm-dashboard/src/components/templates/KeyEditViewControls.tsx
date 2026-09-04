import React from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { CircleHelp } from "lucide-react";

export const labelWithHint = (label: React.ReactNode, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent className="max-w-xs">{hint}</TooltipContent>
    </Tooltip>
  </>
);

const KEY_TYPE_OPTIONS = [
  { value: "default", label: "Full Access", hint: "Can call all routes (AI APIs, Management, and read-only)" },
  { value: "llm_api", label: "AI APIs", hint: "Can call only AI API routes (chat/completions, embeddings, etc.)" },
  { value: "management", label: "Management", hint: "Can call only management routes (user/team/key management)" },
];

export const KeyTypeSelect = ({
  id,
  value,
  onChange,
}: {
  id: string;
  value: string;
  onChange: (value: string) => void;
}) => (
  <Select
    items={Object.fromEntries(KEY_TYPE_OPTIONS.map((option) => [option.value, option.label]))}
    value={value}
    onValueChange={(next: string | null) => next != null && onChange(next)}
  >
    <SelectTrigger id={id} className="w-full">
      <SelectValue placeholder="Select key type" />
    </SelectTrigger>
    <SelectContent>
      {KEY_TYPE_OPTIONS.map((option) => (
        <SelectItem key={option.value} value={option.value}>
          <div className="py-1">
            <div className="font-medium">{option.label}</div>
            <div className="mt-0.5 text-[11px] text-muted-foreground">{option.hint}</div>
          </div>
        </SelectItem>
      ))}
    </SelectContent>
  </Select>
);
