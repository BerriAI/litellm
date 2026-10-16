import CopyButton from "@/components/shared/CopyButton";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { DEFAULT_MAX_WIDTH, FONT_FAMILY_MONO } from "./constants";

interface TruncatedValueProps {
  value?: string;
  maxWidth?: number;
}

/**
 * Displays a truncated value with tooltip and copy functionality.
 * Useful for displaying long IDs, URLs, or other text that may overflow.
 */
export function TruncatedValue({ value, maxWidth = DEFAULT_MAX_WIDTH }: TruncatedValueProps) {
  if (!value) return <span className="text-muted-foreground">-</span>;

  return (
    <TooltipProvider delay={300}>
      <Tooltip>
        <TooltipTrigger
          render={
            <span className="inline-flex items-center gap-1 align-bottom">
              <span className="truncate text-xs" style={{ maxWidth, fontFamily: FONT_FAMILY_MONO }}>
                {value}
              </span>
              <CopyButton value={value} label="Copy" className="size-4 shrink-0" iconClassName="size-3" />
            </span>
          }
        />
        <TooltipContent>{value}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
