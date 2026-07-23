"use client";

import type { ReactNode } from "react";
import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  routerSettingsTabHref,
  slugFromPathname,
  type RouterSettingsTabSlug,
} from "@/app/(dashboard)/router-settings/tabRoutes";

const BASE_TAB_KEY = "loadbalancing";

const ORDERED_KEYS: Array<"" | RouterSettingsTabSlug> = [
  "",
  "routing-groups",
  "fallbacks",
  "prompt-caching",
  "general",
];

const TAB_LABELS: Record<"" | RouterSettingsTabSlug, string> = {
  "": "Loadbalancing",
  "routing-groups": "Routing Groups",
  fallbacks: "Fallbacks",
  "prompt-caching": "Prompt Caching",
  general: "General",
};

export default function RouterSettingsLayout({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();

  const activeSlug = slugFromPathname(pathname);
  const isKnownSlug = ORDERED_KEYS.some((slug) => slug === activeSlug);
  const activeKey = isKnownSlug ? activeSlug || BASE_TAB_KEY : BASE_TAB_KEY;

  useEffect(() => {
    if (activeSlug !== "" && !isKnownSlug) {
      window.location.replace(routerSettingsTabHref(""));
    }
  }, [activeSlug, isKnownSlug]);

  return (
    <div className="w-full">
      <Tabs
        value={activeKey}
        onValueChange={(key) => router.push(routerSettingsTabHref(key === BASE_TAB_KEY ? "" : key))}
        className="px-8 pt-4"
      >
        <TabsList variant="line">
          {ORDERED_KEYS.map((slug) => (
            <TabsTrigger key={slug || BASE_TAB_KEY} value={slug || BASE_TAB_KEY}>
              {TAB_LABELS[slug]}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      <div className="px-8 py-6">{children}</div>
    </div>
  );
}
