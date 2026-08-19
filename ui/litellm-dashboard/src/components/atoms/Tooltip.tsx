import React from "react";
import { CircleHelp } from "lucide-react";

import { Tooltip as ShadcnTooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/cva.config";

interface TooltipProps {
  content: React.ReactNode;
  children?: React.ReactNode;
  width?: string;
  className?: string;
  side?: React.ComponentProps<typeof TooltipContent>["side"];
}

const widthClassNames: Record<string, string> = {
  "360px": "max-w-[360px]",
  "500px": "max-w-[500px]",
  auto: "max-w-xs",
};

const triggerClassName = (className?: string): string =>
  cn(
    "inline-flex cursor-help items-center rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
    className,
  );

const defaultTrigger = <CircleHelp aria-label="question-circle" className="ml-1 size-4 text-muted-foreground" />;

export const Tooltip: React.FC<TooltipProps> = ({ content, children, width = "auto", className, side }) =>
  content === undefined || content === null || content === "" ? (
    <span className={triggerClassName(className)}>{children ?? defaultTrigger}</span>
  ) : (
    <TooltipProvider>
      <ShadcnTooltip>
        <TooltipTrigger render={<span className={triggerClassName(className)} />}>
          {children ?? defaultTrigger}
        </TooltipTrigger>
        <TooltipContent side={side} className={cn("whitespace-normal", widthClassNames[width] ?? "max-w-xs")}>
          {content}
        </TooltipContent>
      </ShadcnTooltip>
    </TooltipProvider>
  );
