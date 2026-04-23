import React, { useState } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Check, Copy } from "lucide-react";
import {
  DEFAULT_MAX_WIDTH,
  FONT_FAMILY_MONO,
  FONT_SIZE_SMALL,
} from "./constants";

interface TruncatedValueProps {
  value?: string;
  maxWidth?: number;
}

/**
 * Displays a truncated value with hover tooltip and inline copy button.
 */
export function TruncatedValue({
  value,
  maxWidth = DEFAULT_MAX_WIDTH,
}: TruncatedValueProps) {
  const [copied, setCopied] = useState(false);
  if (!value) return <span className="text-muted-foreground">-</span>;

  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (_err) {
      /* noop */
    }
  };

  return (
    <span className="inline-flex items-center gap-1 align-bottom">
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <span
              className="inline-block truncate"
              style={{
                maxWidth,
                fontFamily: FONT_FAMILY_MONO,
                fontSize: FONT_SIZE_SMALL,
              }}
            >
              {value}
            </span>
          </TooltipTrigger>
          <TooltipContent className="max-w-md break-all">
            {value}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <TooltipProvider>
        <Tooltip open={copied || undefined}>
          <TooltipTrigger asChild>
            <button
              type="button"
              onClick={handleCopy}
              className="text-muted-foreground hover:text-foreground"
              aria-label="Copy"
            >
              {copied ? (
                <Check className="h-3 w-3 text-emerald-500" />
              ) : (
                <Copy className="h-3 w-3" />
              )}
            </button>
          </TooltipTrigger>
          <TooltipContent>{copied ? "Copied!" : "Copy"}</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </span>
  );
}
