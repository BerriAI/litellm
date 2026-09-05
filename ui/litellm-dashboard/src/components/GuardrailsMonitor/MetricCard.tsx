import { CircleHelp } from "lucide-react";
import React, { type ReactNode } from "react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface MetricCardProps {
  label: string;
  value: string | number;
  valueColor?: string;
  icon?: ReactNode;
  subtitle?: string;
  hint?: ReactNode;
}

export function MetricCard({ label, value, valueColor = "text-foreground", icon, subtitle, hint }: MetricCardProps) {
  return (
    <div role="group" aria-label={label} className="h-full bg-card border border-border rounded-lg p-5 flex flex-col">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium text-muted-foreground">{label}</span>
        {icon && <span className="text-muted-foreground">{icon}</span>}
      </div>
      <div className={`text-3xl font-semibold ${valueColor} tracking-tight`}>{value}</div>
      {subtitle && <p className="text-xs text-muted-foreground mt-1">{subtitle}</p>}
      {hint && (
        <TooltipProvider delay={300}>
          <Tooltip>
            <TooltipTrigger
              render={
                <button
                  type="button"
                  className="mt-2 inline-flex w-fit cursor-help items-start gap-1 text-left text-xs text-muted-foreground hover:text-foreground"
                >
                  <CircleHelp className="mt-px size-3.5 shrink-0" />
                  How is this calculated?
                </button>
              }
            />
            <TooltipContent side="bottom" align="start" className="max-w-sm">
              {hint}
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  );
}
