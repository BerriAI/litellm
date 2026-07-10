"use client";

import { Meter as MeterPrimitive } from "@base-ui/react/meter";
import { type VariantProps } from "cva";
import * as React from "react";

import { cn, cva } from "@/lib/cva.config";

const meterIndicatorVariants = cva({
  base: "h-full rounded-full transition-[width] duration-300",
  variants: {
    tone: {
      default: "bg-primary",
      warning: "bg-amber-500",
      over: "bg-destructive",
    },
  },
  defaultVariants: { tone: "default" },
});

const Meter = React.forwardRef<React.ComponentRef<typeof MeterPrimitive.Root>, MeterPrimitive.Root.Props>(
  ({ className, ...props }, ref) => (
    <MeterPrimitive.Root
      ref={ref}
      data-slot="meter"
      className={cn("flex w-full flex-col gap-1.5", className)}
      {...props}
    />
  ),
);
Meter.displayName = "Meter";

const MeterLabel = React.forwardRef<React.ComponentRef<typeof MeterPrimitive.Label>, MeterPrimitive.Label.Props>(
  ({ className, ...props }, ref) => (
    <MeterPrimitive.Label
      ref={ref}
      data-slot="meter-label"
      className={cn("text-xs text-muted-foreground", className)}
      {...props}
    />
  ),
);
MeterLabel.displayName = "MeterLabel";

const MeterValue = React.forwardRef<React.ComponentRef<typeof MeterPrimitive.Value>, MeterPrimitive.Value.Props>(
  ({ className, ...props }, ref) => (
    <MeterPrimitive.Value
      ref={ref}
      data-slot="meter-value"
      className={cn("text-xs font-medium tabular-nums", className)}
      {...props}
    />
  ),
);
MeterValue.displayName = "MeterValue";

const MeterTrack = React.forwardRef<React.ComponentRef<typeof MeterPrimitive.Track>, MeterPrimitive.Track.Props>(
  ({ className, ...props }, ref) => (
    <MeterPrimitive.Track
      ref={ref}
      data-slot="meter-track"
      className={cn("h-1.5 w-full overflow-hidden rounded-full bg-muted", className)}
      {...props}
    />
  ),
);
MeterTrack.displayName = "MeterTrack";

const MeterIndicator = React.forwardRef<
  React.ComponentRef<typeof MeterPrimitive.Indicator>,
  MeterPrimitive.Indicator.Props & VariantProps<typeof meterIndicatorVariants>
>(({ className, tone, ...props }, ref) => (
  <MeterPrimitive.Indicator
    ref={ref}
    data-slot="meter-indicator"
    className={cn(meterIndicatorVariants({ tone, className }))}
    {...props}
  />
));
MeterIndicator.displayName = "MeterIndicator";

export { Meter, MeterLabel, MeterValue, MeterTrack, MeterIndicator };
