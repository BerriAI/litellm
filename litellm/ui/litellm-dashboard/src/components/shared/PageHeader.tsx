"use client";

import type { ComponentProps, ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { cn } from "@/lib/cva.config";

type PageHeaderProps = ComponentProps<"div"> & { aside?: ReactNode };

export function PageHeader({ aside, className, children, ...props }: PageHeaderProps) {
  if (!aside) {
    return (
      <div data-slot="page-header" className={cn("shrink-0", className)} {...props}>
        {children}
      </div>
    );
  }
  return (
    <div
      data-slot="page-header"
      className={cn("flex shrink-0 flex-wrap items-start justify-between gap-x-6 gap-y-3", className)}
      {...props}
    >
      <div className="min-w-0 flex-[1_1_16rem]">{children}</div>
      <div data-slot="page-header-aside" className="flex min-h-8 flex-wrap items-center gap-2">
        {aside}
      </div>
    </div>
  );
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
  const { t } = useTranslation();
  return (
    <div
      role="group"
      aria-label={t("shell.pageHeader.controls")}
      className={cn("mt-5 flex min-h-9 flex-wrap items-center gap-2", className)}
      {...props}
    />
  );
}
