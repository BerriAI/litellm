"use client";

import * as React from "react";

import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface CellTooltipProps {
  content: React.ReactNode;
  trigger: React.ReactElement;
}

export function CellTooltip({ content, trigger }: CellTooltipProps) {
  return (
    <TooltipProvider delay={300}>
      <Tooltip>
        <TooltipTrigger render={trigger} />
        <TooltipContent>{content}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
