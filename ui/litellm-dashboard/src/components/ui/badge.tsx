import * as React from "react";
import { type VariantProps } from "cva";

import { cn, cva } from "@/lib/cva.config";

const badgeVariants = cva({
  base: "inline-flex w-fit shrink-0 items-center justify-center gap-1 overflow-hidden rounded-full border border-transparent px-2 py-0.5 text-xs font-medium whitespace-nowrap transition-[color,box-shadow] focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 [&>svg]:pointer-events-none [&>svg]:size-3",
  variants: {
    variant: {
      default: "bg-primary text-primary-foreground [a&]:hover:bg-primary/90",
      secondary: "bg-secondary text-secondary-foreground [a&]:hover:bg-secondary/90",
      destructive:
        "bg-destructive text-white focus-visible:ring-destructive/20 dark:bg-destructive/60 dark:focus-visible:ring-destructive/40 [a&]:hover:bg-destructive/90",
      outline: "border-border text-foreground [a&]:hover:bg-accent [a&]:hover:text-accent-foreground",
      ghost: "[a&]:hover:bg-accent [a&]:hover:text-accent-foreground",
      link: "text-primary underline-offset-4 [a&]:hover:underline",
    },
  },
  defaultVariants: {
    variant: "default",
  },
});

const Badge = React.forwardRef<
  HTMLSpanElement,
  React.ComponentPropsWithoutRef<"span"> & VariantProps<typeof badgeVariants>
>(({ className, variant = "default", ...props }, ref) => (
  <span
    ref={ref}
    data-slot="badge"
    data-variant={variant}
    className={cn(badgeVariants({ variant }), className)}
    {...props}
  />
));
Badge.displayName = "Badge";

export { Badge, badgeVariants };
