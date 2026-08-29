import type * as React from "react";

import { Alert as AlertPrimitive, AlertAction, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { cn } from "@/lib/cva.config";

const STATUS_VARIANT_CLASSES = {
  info: "border-info/20 bg-info/5 text-info *:[svg]:text-current",
  success: "border-success/20 bg-success/5 text-success *:[svg]:text-current",
  warning: "border-warning/20 bg-warning/5 text-warning *:[svg]:text-current",
  error:
    "border-destructive/20 bg-destructive/10 text-destructive *:data-[slot=alert-description]:text-destructive/90 *:[svg]:text-destructive",
} as const;

type StatusVariant = keyof typeof STATUS_VARIANT_CLASSES;
type AlertVariant = NonNullable<React.ComponentProps<typeof AlertPrimitive>["variant"]> | StatusVariant;

type AlertProps = Omit<React.ComponentProps<typeof AlertPrimitive>, "variant"> & {
  variant?: AlertVariant;
};

const isStatusVariant = (variant: AlertVariant): variant is StatusVariant => variant in STATUS_VARIANT_CLASSES;

const Alert = ({ variant = "default", className, ...props }: AlertProps) => (
  <AlertPrimitive
    data-variant={variant}
    variant={variant === "destructive" ? "destructive" : "default"}
    className={cn(isStatusVariant(variant) ? STATUS_VARIANT_CLASSES[variant] : undefined, className)}
    {...props}
  />
);

export { Alert, AlertTitle, AlertDescription, AlertAction };
