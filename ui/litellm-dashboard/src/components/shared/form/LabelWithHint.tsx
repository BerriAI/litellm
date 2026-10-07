"use client";

import { CircleHelp } from "lucide-react";
import React from "react";

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

const hintIconClassName = "size-3.5 shrink-0 cursor-help text-muted-foreground";

export const labelWithHint = (label: React.ReactNode, hint: React.ReactNode): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className={hintIconClassName} />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

export const labelWithDocsHint = (label: React.ReactNode, hint: React.ReactNode, href: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger
        render={
          <a href={href} target="_blank" rel="noopener noreferrer" onClick={(event) => event.stopPropagation()}>
            <CircleHelp className={hintIconClassName} />
          </a>
        }
      />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);
