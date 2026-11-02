import type { ReactNode } from "react";
import { Check } from "lucide-react";

export function Panel({
  title,
  extra,
  children,
  className = "",
}: {
  title: ReactNode;
  extra?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`overflow-hidden rounded-xl border border-border bg-card ${className}`}>
      <div className="flex min-h-12 items-center justify-between gap-2 border-b border-border px-4 py-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold tracking-tight">{title}</h2>
        {extra}
      </div>
      {children}
    </section>
  );
}

export function Eyebrow({ children }: { children: ReactNode }) {
  return <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">{children}</p>;
}

export function Success({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400">
      <Check className="size-3.5" />
      {children}
    </span>
  );
}

export function Note({ icon, children }: { icon: ReactNode; children: ReactNode }) {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-border bg-muted/40 p-4 text-xs leading-relaxed text-muted-foreground">
      <span className="mt-0.5 shrink-0">{icon}</span>
      <div>{children}</div>
    </div>
  );
}
