"use client";

import React from "react";
import { Info } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

export interface SummaryCardProps {
  label: string;
  value: string;
  hint?: string;
  /** Rendered behind an info affordance. Use it for how a figure is derived, not for restating the label. */
  info?: string;
  /** A related figure the headline is a share of, shown beside it. */
  secondary?: { label: string; value: string };
}

/**
 * A labelled figure with an optional hint line and an optional "how is this calculated" popover.
 * Shared by the proxy-wide Cost Optimization usage tab and the per-key savings tab so both
 * surfaces present the same figures identically.
 */
const slugOf = (label: string): string => label.toLowerCase().replace(/\s+/g, "-");

const SummaryCard = ({ label, value, hint, info, secondary }: SummaryCardProps) => (
  <Card data-testid={`summary-card-${slugOf(label)}`}>
    <CardHeader className="flex flex-row items-center justify-between space-y-0">
      <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
      {info && (
        <Popover>
          <PopoverTrigger
            aria-label={`How ${label.toLowerCase()} is calculated`}
            data-testid={`summary-card-info-${slugOf(label)}`}
            className="cursor-pointer text-muted-foreground hover:text-foreground"
          >
            <Info className="size-3.5" />
          </PopoverTrigger>
          <PopoverContent align="end" className="w-64 text-sm text-muted-foreground">
            {info}
          </PopoverContent>
        </Popover>
      )}
    </CardHeader>
    <CardContent>
      <div className="flex items-end gap-4">
        <div>
          <p className="text-2xl font-semibold text-foreground">{value}</p>
          {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
        </div>
        {secondary && (
          <div className="self-stretch border-l pl-4">
            <div className="flex h-full flex-col justify-end">
              <p className="text-lg font-medium text-muted-foreground">{secondary.value}</p>
              <p className="mt-1 text-xs text-muted-foreground">{secondary.label}</p>
            </div>
          </div>
        )}
      </div>
    </CardContent>
  </Card>
);

export default SummaryCard;
