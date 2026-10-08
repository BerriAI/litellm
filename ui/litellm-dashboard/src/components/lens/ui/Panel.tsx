"use client";

import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

import { cn } from "@/lib/cva.config";

export function Panel({
  icon: Icon,
  title,
  subtitle,
  action,
  className,
  children,
}: {
  icon?: LucideIcon;
  title?: ReactNode;
  subtitle?: ReactNode;
  action?: ReactNode;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={cn("flex min-h-0 min-w-0 flex-col overflow-hidden rounded-xl border bg-card", className)}>
      {(title || action) && (
        <header className="flex min-h-11 shrink-0 flex-wrap items-center justify-between gap-x-3 gap-y-2 px-4 pt-3 pb-2">
          <div className="min-w-0">
            {title && (
              <h2 className="flex min-w-0 items-center gap-2 text-sm font-medium text-foreground">
                {Icon && (
                  <Icon aria-hidden="true" className="size-4 shrink-0 text-muted-foreground" strokeWidth={1.75} />
                )}
                <span className="truncate">{title}</span>
              </h2>
            )}
            {subtitle && <p className="mt-0.5 truncate text-xs text-muted-foreground">{subtitle}</p>}
          </div>
          {action && <div className="flex flex-wrap items-center gap-2">{action}</div>}
        </header>
      )}
      {children}
    </section>
  );
}
