"use client";

import type { LucideIcon } from "lucide-react";
import { useId, type CSSProperties, type ReactNode } from "react";

import { cn } from "@/lib/cva.config";

const LINE_STRUT = "​";

interface SectionCardProps {
  icon: LucideIcon;
  title: string;
  action?: ReactNode;
  footer?: ReactNode;
  children: ReactNode;
  className?: string;
  testId?: string;
}

export function SectionCard({ icon: Icon, title, action, footer, children, className, testId }: SectionCardProps) {
  const headingId = useId();
  return (
    <section
      aria-labelledby={headingId}
      data-testid={testId}
      className={cn(
        "flex min-w-0 flex-col gap-4 rounded-xl border border-border bg-card p-4 text-[13px] leading-5 text-card-foreground",
        className,
      )}
    >
      <div className="flex items-start gap-2">
        <h2 id={headingId} className="m-0 flex flex-1 items-center gap-2 text-[15px] leading-[22px] font-semibold">
          <Icon className="size-4 shrink-0 text-muted-foreground" aria-hidden />
          {title}
        </h2>
        {action}
      </div>
      {children}
      {footer ? <div className="flex flex-col gap-1.5 border-t border-border pt-4">{footer}</div> : null}
    </section>
  );
}

export function SubHead({ title, count }: { title: string; count?: number }) {
  return (
    <div className="flex items-center gap-1.5 font-medium">
      {title}
      {count !== undefined ? <CountPill value={count} /> : null}
    </div>
  );
}

export function CountPill({ value }: { value: number }) {
  return (
    <span className="inline-flex h-[18px] items-center rounded-full bg-muted px-1.5 text-xs font-medium text-muted-foreground tabular-nums">
      {value}
    </span>
  );
}

export function FactLabel({ children }: { children: ReactNode }) {
  return <div className="text-xs font-medium text-muted-foreground">{children}</div>;
}

export function Caption({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("text-xs text-muted-foreground", className)}>{children}</div>;
}

export function ValueSkeleton({ width, height = 12 }: { width: number; height?: number }) {
  const style: CSSProperties = { width, height };
  return (
    <span className="inline-flex items-center" aria-hidden>
      {LINE_STRUT}
      <span data-slot="skeleton" className="inline-block animate-pulse rounded-md bg-muted" style={style} />
    </span>
  );
}

export function BadgeRow({ children }: { children: ReactNode }) {
  return <div className="flex flex-wrap gap-1">{children}</div>;
}
