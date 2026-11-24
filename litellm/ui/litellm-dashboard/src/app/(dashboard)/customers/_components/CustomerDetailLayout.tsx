import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { cn } from "@/lib/cva.config";

export function CustomerDetailLayout({ children, label }: { children: ReactNode; label?: string }) {
  return (
    <main className="space-y-6" aria-label={label}>
      {children}
    </main>
  );
}

export function CustomerDetailHeader({
  title,
  breadcrumbs,
  breadcrumbLabel,
  children,
  actions,
}: {
  title: ReactNode;
  breadcrumbs: ReactNode;
  breadcrumbLabel?: string;
  children?: ReactNode;
  actions?: ReactNode;
}) {
  const { t } = useTranslation();
  return (
    <header>
      <nav
        aria-label={breadcrumbLabel ?? t("customerPortal.analytics.breadcrumb")}
        className="mb-5 flex flex-wrap items-center gap-2 text-xs text-muted-foreground"
      >
        {breadcrumbs}
      </nav>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0 flex-1 basis-72">
          <h1 className="break-words text-2xl font-semibold tracking-tight">{title}</h1>
          {children}
        </div>
        {actions}
      </div>
    </header>
  );
}

export function CustomerDetailNavigation({
  label,
  items,
}: {
  label: string;
  items: readonly { href: string; label: string; active: boolean }[];
}) {
  return (
    <nav aria-label={label} className="flex flex-wrap gap-x-6 gap-y-3 border-b">
      {items.map((item) => (
        <a
          key={item.href}
          href={item.href}
          aria-current={item.active ? "page" : undefined}
          className={cn(
            "shrink-0 whitespace-nowrap border-b-2 border-transparent px-0.5 pb-3 text-sm font-medium text-muted-foreground hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring",
            item.active && "border-primary text-foreground",
          )}
        >
          {item.label}
        </a>
      ))}
    </nav>
  );
}
