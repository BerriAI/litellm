"use client";

import { Check, Copy } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

const COPIED_RESET_MS = 1600;

interface CopyButtonProps {
  value: string;
  label?: string;
  copiedLabel?: string;
  iconOnly?: boolean;
  className?: string;
}

/** Copy → check for a moment. `iconOnly` renders a bare icon button with a tooltip. */
export function CopyButton({
  value,
  label = "Copy",
  copiedLabel = "Copied",
  iconOnly = false,
  className,
}: CopyButtonProps) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const timeout = window.setTimeout(() => setCopied(false), COPIED_RESET_MS);
    return () => window.clearTimeout(timeout);
  }, [copied]);

  const button = (
    <Button
      variant={iconOnly ? "ghost" : "outline"}
      size={iconOnly ? "icon-xs" : "xs"}
      className={cn(
        "font-mono text-[10px] text-muted-foreground hover:text-foreground",
        !iconOnly && "gap-1.5",
        className,
      )}
      onClick={async () => setCopied(await copyToClipboard(value, copiedLabel))}
      aria-label={label}
    >
      {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
      {!iconOnly && <span>{copied ? copiedLabel : label}</span>}
    </Button>
  );

  if (!iconOnly) return button;
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger render={button} />
        <TooltipContent>{copied ? copiedLabel : label}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
