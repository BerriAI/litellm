import type { ComponentProps } from "react";
import { cn } from "@/lib/cva.config";

export type SettingsSectionProps = ComponentProps<"section"> & { heading: string; description: string };

export function SettingsSection({ heading, description, className, children, ...props }: SettingsSectionProps) {
  return (
    <section
      data-slot="settings-section"
      aria-label={heading}
      {...props}
      className={cn("grid gap-4 py-6 first:pt-0 last:pb-0 md:grid-cols-[200px_minmax(0,1fr)] md:gap-8", className)}
    >
      <div className="space-y-1">
        <h2 className="text-sm font-semibold">{heading}</h2>
        <p className="text-xs leading-5 text-muted-foreground">{description}</p>
      </div>
      <div className="min-w-0 space-y-3">{children}</div>
    </section>
  );
}

export type SettingsCardProps = ComponentProps<"div">;

export function SettingsCard({ className, ...props }: SettingsCardProps) {
  return (
    <div
      data-slot="settings-card"
      {...props}
      className={cn("rounded-lg border border-border bg-card p-4", className)}
    />
  );
}
