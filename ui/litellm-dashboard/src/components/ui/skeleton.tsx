import * as React from "react";

import { cn } from "@/lib/cva.config";

const Skeleton = React.forwardRef<HTMLDivElement, React.ComponentPropsWithoutRef<"div">>(
  ({ className, ...props }, ref) => (
    <div ref={ref} data-slot="skeleton" className={cn("animate-pulse rounded-md bg-accent", className)} {...props} />
  ),
);
Skeleton.displayName = "Skeleton";

export { Skeleton };
