"use client";

import type { ReactNode } from "react";
import { routerSettingsRoutes } from "@/app/(dashboard)/router-settings/tabRoutes";
import { useTabRouting } from "@/app/(dashboard)/hooks/useTabRouting";
import { TabRouteBar } from "@/app/(dashboard)/components/TabRouteBar";

const BASE_TAB_KEY = "loadbalancing";

const TABS = [
  { key: BASE_TAB_KEY, label: "Loadbalancing" },
  { key: "routing-groups", label: "Routing Groups" },
  { key: "fallbacks", label: "Fallbacks" },
  { key: "prompt-caching", label: "Prompt Caching" },
  { key: "general", label: "General" },
] as const;

export default function RouterSettingsLayout({ children }: { children: ReactNode }) {
  const { activeKey } = useTabRouting({
    routes: routerSettingsRoutes,
    baseTabKey: BASE_TAB_KEY,
    visibleKeys: routerSettingsRoutes.slugs,
  });

  return (
    <div className="w-full">
      <TabRouteBar
        routes={routerSettingsRoutes}
        baseTabKey={BASE_TAB_KEY}
        activeKey={activeKey}
        tabs={TABS}
        className="px-8 pt-4"
      />
      <div className="px-8 py-6">{children}</div>
    </div>
  );
}
