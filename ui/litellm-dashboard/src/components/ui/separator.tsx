"use client";

import { Separator as SeparatorPrimitive } from "@base-ui/react/separator";
import * as React from "react";

import { cn } from "@/lib/cva.config";

const Separator = React.forwardRef<React.ComponentRef<typeof SeparatorPrimitive>, SeparatorPrimitive.Props>(
  ({ className, orientation = "horizontal", ...props }, ref) => (
    <SeparatorPrimitive
      ref={ref}
      data-slot="separator"
      orientation={orientation}
      className={cn(
        "shrink-0 bg-border data-horizontal:h-px data-horizontal:w-full data-vertical:w-px data-vertical:self-stretch",
        className,
      )}
      {...props}
    />
  ),
);
Separator.displayName = "Separator";

export { Separator };
