"use client";

import { Tag } from "antd";

import CopyButton from "@/components/shared/CopyButton";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/cva.config";

interface CopyableBadgeProps {
  value: string;
  color?: string;
  maxWidthClassName?: string;
  dataTestId?: string;
}

export function CopyableBadge({
  value,
  color = "red",
  maxWidthClassName = "max-w-[130px]",
  dataTestId,
}: CopyableBadgeProps) {
  return (
    <Tag color={color} className="m-0 inline-flex max-w-full items-center gap-1" data-testid={dataTestId}>
      <Tooltip>
        <TooltipTrigger className="max-w-full cursor-default border-0 bg-transparent p-0">
          <span className={cn("block truncate", maxWidthClassName)}>{value}</span>
        </TooltipTrigger>
        <TooltipContent className="max-w-sm">
          <span className="break-all">{value}</span>
        </TooltipContent>
      </Tooltip>
      <CopyButton value={value} label={`Copy ${value}`} iconClassName="size-3" />
    </Tag>
  );
}
