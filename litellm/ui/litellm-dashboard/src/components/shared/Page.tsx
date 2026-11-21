"use client";

import type { ComponentProps } from "react";
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
      className={cn("h-10 w-full shrink-0 justify-start gap-6 overflow-x-auto border-b border-border p-0", className)}
      {...props}
    />
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
