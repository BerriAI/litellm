"use client";

import type { ReactNode } from "react";
import { cachingRoutes } from "@/app/(dashboard)/caching/tabRoutes";
import { useTabRouting } from "@/app/(dashboard)/hooks/useTabRouting";
import { TabRouteBar } from "@/app/(dashboard)/components/TabRouteBar";

const BASE_TAB_KEY = "cache-analytics";

const TABS = [
  { key: BASE_TAB_KEY, label: "Cache Analytics" },
  { key: "health", label: "Cache Health" },
  { key: "settings", label: "Cache Settings" },
  { key: "coordination-redis", label: "Coordination Redis" },
] as const;

export default function CachingLayout({ children }: { children: ReactNode }) {
  const { activeKey } = useTabRouting({
    routes: cachingRoutes,
    baseTabKey: BASE_TAB_KEY,
    visibleKeys: cachingRoutes.slugs,
  });

  return (
    <div className="p-8 w-full mt-2 mb-8">
      <TabRouteBar routes={cachingRoutes} baseTabKey={BASE_TAB_KEY} activeKey={activeKey} tabs={TABS} />
      <div className="mt-4">{children}</div>
    </div>
  );
}
