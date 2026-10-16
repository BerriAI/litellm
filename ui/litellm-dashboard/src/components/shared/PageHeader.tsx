"use client";

import type { ComponentProps } from "react";

import { cn } from "@/lib/cva.config";

export function PageHeader({ className, ...props }: ComponentProps<"div">) {
  return <div className={cn("shrink-0", className)} {...props} />;
}

export function PageHeaderTitle({ className, ...props }: ComponentProps<"h1">) {
  return (
    <h1
      className={cn(
        "flex items-center gap-2.5 text-2xl font-semibold tracking-tight text-foreground [&_svg]:size-5 [&_svg]:flex-none [&_svg]:stroke-[1.75]",
        className,
      )}
      {...props}
    />
  );
}

export function PageHeaderDescription({ className, ...props }: ComponentProps<"p">) {
  return <p className={cn("mt-1.5 text-sm text-muted-foreground", className)} {...props} />;
}

export function PageHeaderControls({ className, ...props }: ComponentProps<"div">) {
  return (
    <div
      role="group"
      aria-label="Page controls"
      className={cn("mt-5 flex min-h-9 items-center gap-2", className)}
      {...props}
    />
  );
}
