"use client";

import type { ReactNode } from "react";
import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cacheTabHref, slugFromPathname, type CacheTabSlug } from "@/app/(dashboard)/caching/tabRoutes";

const BASE_TAB_KEY = "cache-analytics";

const ORDERED_KEYS: Array<"" | CacheTabSlug> = ["", "health", "settings", "coordination-redis"];

const TAB_LABELS: Record<"" | CacheTabSlug, string> = {
  "": "Cache Analytics",
  health: "Cache Health",
  settings: "Cache Settings",
  "coordination-redis": "Coordination Redis",
};

export default function CachingLayout({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();

  const activeSlug = slugFromPathname(pathname);
  const isKnownSlug = ORDERED_KEYS.some((slug) => slug === activeSlug);
  const activeKey = isKnownSlug ? activeSlug || BASE_TAB_KEY : BASE_TAB_KEY;

  useEffect(() => {
    if (activeSlug !== "" && !isKnownSlug) {
      window.location.replace(cacheTabHref(""));
    }
  }, [activeSlug, isKnownSlug]);

  return (
    <div className="p-8 w-full mt-2 mb-8">
      <Tabs value={activeKey} onValueChange={(key) => router.push(cacheTabHref(key === BASE_TAB_KEY ? "" : key))}>
        <TabsList variant="line">
          {ORDERED_KEYS.map((slug) => (
            <TabsTrigger key={slug || BASE_TAB_KEY} value={slug || BASE_TAB_KEY}>
              {TAB_LABELS[slug]}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      <div className="mt-4">{children}</div>
    </div>
  );
}
