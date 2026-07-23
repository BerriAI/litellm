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
  maxWidthClassName = "max-w-[220px]",
  dataTestId,
}: CopyableBadgeProps) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Tag color={color} className="m-0 max-w-full cursor-default" data-testid={dataTestId}>
            <span className={cn("block truncate", maxWidthClassName)}>{value}</span>
          </Tag>
        }
      />
      <TooltipContent className="max-w-sm">
        <span className="break-all">{value}</span>
        <CopyButton
          value={value}
          label={`Copy ${value}`}
          iconClassName="size-3"
          className="text-background/70 hover:bg-transparent hover:text-background"
        />
      </TooltipContent>
    </Tooltip>
  );
}
