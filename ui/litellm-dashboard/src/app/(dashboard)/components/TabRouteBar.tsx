"use client";

import type { MouseEvent } from "react";
import { useRouter } from "next/navigation";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { TabRoutes } from "@/utils/tabRoutes";

export interface TabRouteItem {
  key: string;
  label: string;
}

interface TabRouteBarProps {
  routes: Pick<TabRoutes<string>, "tabHref">;
  baseTabKey: string;
  activeKey: string;
  tabs: readonly TabRouteItem[];
  className?: string;
}

export function TabRouteBar({ routes, baseTabKey, activeKey, tabs, className }: TabRouteBarProps) {
  const router = useRouter();

  const navigate = (href: string) => (event: MouseEvent<HTMLAnchorElement>) => {
    const commandModifier = event.metaKey || event.ctrlKey;
    const otherModifier = event.shiftKey || event.altKey;
    if (commandModifier || otherModifier) {
      return;
    }
    event.preventDefault();
    router.push(href);
  };

  return (
    <Tabs value={activeKey} className={className}>
      <TabsList variant="line">
        {tabs.map(({ key, label }) => {
          const href = routes.tabHref(key === baseTabKey ? "" : key);
          return (
            <TabsTrigger key={key} value={key} render={<a href={href} onClick={navigate(href)} />}>
              {label}
            </TabsTrigger>
          );
        })}
      </TabsList>
    </Tabs>
  );
}
