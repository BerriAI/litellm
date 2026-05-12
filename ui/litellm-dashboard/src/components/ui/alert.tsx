import * as React from "react";
import { AlertCircle, AlertTriangle, Info, type LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

type AlertVariant = "info" | "warning" | "destructive";

export interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: AlertVariant;
}

const variantClasses: Record<AlertVariant, string> = {
  info: "border-blue-200 bg-blue-50 text-blue-950",
  warning: "border-amber-200 bg-amber-50 text-amber-950",
  destructive: "border-red-200 bg-red-50 text-red-950",
};

const iconClasses: Record<AlertVariant, string> = {
  info: "text-blue-600",
  warning: "text-amber-600",
  destructive: "text-red-600",
};

const icons: Record<AlertVariant, LucideIcon> = {
  info: Info,
  warning: AlertTriangle,
  destructive: AlertCircle,
};

const Alert = React.forwardRef<HTMLDivElement, AlertProps>(({ className, variant = "info", children, ...props }, ref) => {
  const Icon = icons[variant];

  return (
    <div
      ref={ref}
      role="alert"
      className={cn("grid grid-cols-[1rem_1fr] gap-x-3 rounded-lg border p-4 text-sm", variantClasses[variant], className)}
      {...props}
    >
      <Icon aria-hidden="true" className={cn("mt-0.5 size-4", iconClasses[variant])} />
      <div className="min-w-0">{children}</div>
    </div>
  );
});
Alert.displayName = "Alert";

const AlertTitle = React.forwardRef<HTMLHeadingElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h5 ref={ref} className={cn("mb-1 font-medium leading-none tracking-tight", className)} {...props} />
  ),
);
AlertTitle.displayName = "AlertTitle";

const AlertDescription = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => <div ref={ref} className={cn("text-sm leading-relaxed", className)} {...props} />,
);
AlertDescription.displayName = "AlertDescription";

export { Alert, AlertDescription, AlertTitle };
