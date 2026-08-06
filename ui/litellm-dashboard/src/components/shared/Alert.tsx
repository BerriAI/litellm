import * as React from "react";
import { type VariantProps } from "cva";

import { cn, cva } from "@/lib/cva.config";

const alertVariants = cva({
  base: "group/alert relative grid w-full gap-0.5 rounded-lg border px-4 py-3 text-left text-sm has-data-[slot=alert-action]:relative has-data-[slot=alert-action]:pr-18 has-[>svg]:grid-cols-[auto_1fr] has-[>svg]:gap-x-2.5 *:[svg]:row-span-2 *:[svg]:translate-y-0.5 *:[svg]:text-current *:[svg:not([class*='size-'])]:size-4",
  variants: {
    variant: {
      default: "bg-card text-card-foreground",
      destructive: "bg-card text-destructive *:data-[slot=alert-description]:text-destructive/90 *:[svg]:text-current",
      info: "border-blue-200 bg-blue-50 text-blue-900 *:data-[slot=alert-description]:text-blue-800 *:[svg]:text-blue-600",
      warning:
        "border-amber-200 bg-amber-50 text-amber-900 *:data-[slot=alert-description]:text-amber-800 *:[svg]:text-amber-600",
      error: "border-red-200 bg-red-50 text-red-900 *:data-[slot=alert-description]:text-red-800 *:[svg]:text-red-600",
    },
  },
  defaultVariants: {
    variant: "default",
  },
});

type AlertProps = React.ComponentProps<"div"> & VariantProps<typeof alertVariants>;

const Alert = React.forwardRef<HTMLDivElement, AlertProps>(({ className, variant, ...props }, ref) => (
  <div ref={ref} data-slot="alert" role="alert" className={cn(alertVariants({ variant }), className)} {...props} />
));
Alert.displayName = "Alert";

const AlertTitle = React.forwardRef<HTMLDivElement, React.ComponentProps<"div">>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    data-slot="alert-title"
    className={cn(
      "font-medium group-has-[>svg]/alert:col-start-2 [&_a]:underline [&_a]:underline-offset-3 [&_a]:hover:text-foreground",
      className,
    )}
    {...props}
  />
));
AlertTitle.displayName = "AlertTitle";

const AlertDescription = React.forwardRef<HTMLDivElement, React.ComponentProps<"div">>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      data-slot="alert-description"
      className={cn(
        "text-sm text-balance text-muted-foreground md:text-pretty [&_a]:underline [&_a]:underline-offset-3 [&_a]:hover:text-foreground [&_p:not(:last-child)]:mb-4",
        className,
      )}
      {...props}
    />
  ),
);
AlertDescription.displayName = "AlertDescription";

const AlertAction = React.forwardRef<HTMLDivElement, React.ComponentProps<"div">>(({ className, ...props }, ref) => (
  <div ref={ref} data-slot="alert-action" className={cn("absolute top-2.5 right-3", className)} {...props} />
));
AlertAction.displayName = "AlertAction";

export { Alert, AlertTitle, AlertDescription, AlertAction };
