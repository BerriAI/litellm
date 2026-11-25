"use client";

import type { ComponentProps, ReactNode } from "react";
import { ToolbarSeparator } from "@/components/shared/ToolbarSeparator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/cva.config";

export function Page({ className, ...props }: ComponentProps<"div">) {
  return <div className={cn("flex min-h-0 min-w-0 w-full flex-col gap-6 p-4 sm:p-6", className)} {...props} />;
}

export function PageContent({ className, ...props }: ComponentProps<"div">) {
  return <div className={cn("flex min-h-0 min-w-0 flex-1 flex-col gap-6", className)} {...props} />;
}

export function PageTabs({ className, ...props }: ComponentProps<typeof Tabs>) {
  return <Tabs className={cn("min-h-0 min-w-0 flex-1 gap-6", className)} {...props} />;
}

export function PageTabsList({ className, ...props }: ComponentProps<typeof TabsList>) {
  return (
    <TabsList
      variant="line"
      className={cn(
        "no-scrollbar scroll-fade-e h-9 w-full shrink-0 justify-start gap-6 overflow-x-auto border-b border-border p-0",
        className,
      )}
      {...props}
    />
  );
}

type PageTabsBarProps = ComponentProps<typeof TabsList> & { actions?: ReactNode };

export function PageTabsBar({ actions, className, ...props }: PageTabsBarProps) {
  if (!actions) return <PageTabsList className={className} {...props} />;
  return (
    <div data-slot="page-tabs-bar" className="@container w-full min-w-0">
      <div className="flex flex-col gap-3 @3xl:flex-row @3xl:gap-0">
        <div className="flex flex-wrap items-center gap-2 @3xl:h-9 @3xl:shrink-0 @3xl:flex-nowrap @3xl:border-b @3xl:border-border @3xl:pr-6">
          {actions}
        </div>
        <div className="hidden h-9 shrink-0 items-center border-b border-border pr-6 @3xl:flex">
          <ToolbarSeparator className="mx-0 h-6" />
        </div>
        <PageTabsList className={cn("min-w-0 flex-1", className)} {...props} />
      </div>
    </div>
  );
}

export function PageTabsTrigger({ className, ...props }: ComponentProps<typeof TabsTrigger>) {
  return (
    <TabsTrigger
      className={cn("flex-none px-0 font-normal group-data-horizontal/tabs:after:bottom-0", className)}
      {...props}
    />
  );
}

export function PageTabsContent({ className, ...props }: ComponentProps<typeof TabsContent>) {
  return <TabsContent className={cn("flex min-h-0 min-w-0 flex-1 flex-col", className)} {...props} />;
}
