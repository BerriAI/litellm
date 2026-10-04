import type { ReactNode } from "react";
import { cn } from "@/lib/cva.config";

export function SettingsSection({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <section
      aria-label={title}
      className="grid gap-4 py-6 first:pt-0 last:pb-0 md:grid-cols-[200px_minmax(0,1fr)] md:gap-8"
    >
      <div className="space-y-1">
        <h2 className="text-sm font-semibold">{title}</h2>
        <p className="text-xs leading-5 text-muted-foreground">{description}</p>
      </div>
      <div className="min-w-0 space-y-3">{children}</div>
    </section>
  );
}

export function SettingsCard({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn("rounded-lg border border-border bg-card p-4", className)}>{children}</div>;
}
