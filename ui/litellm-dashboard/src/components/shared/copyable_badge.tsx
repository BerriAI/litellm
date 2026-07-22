"use client";

import { Tag } from "antd";
import { Check, Copy } from "lucide-react";
import * as React from "react";

import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

interface CopyableBadgeProps {
  value: string;
  color?: string;
  maxWidthClassName?: string;
  dataTestId?: string;
}

export function CopyableBadge({
  value,
  color = "red",
  maxWidthClassName = "max-w-[220px]",
  dataTestId,
}: CopyableBadgeProps) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    const success = await copyToClipboard(value);
    if (success) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <TooltipProvider delay={300}>
      <Tooltip>
        <TooltipTrigger
          aria-label={`Copy ${value}`}
          data-testid={dataTestId}
          onClick={handleCopy}
          className="inline-flex max-w-full cursor-pointer border-0 bg-transparent p-0"
        >
          <Tag color={color} className="m-0 max-w-full">
            <span className={cn("block truncate", maxWidthClassName)}>{value}</span>
          </Tag>
        </TooltipTrigger>
        <TooltipContent>
          <span className="inline-flex items-center gap-1.5">
            {copied ? <Check className="size-3 shrink-0" /> : <Copy className="size-3 shrink-0" />}
            <span className="font-mono break-all">{value}</span>
            <span className="opacity-70">{copied ? "Copied" : "Click to copy"}</span>
          </span>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
