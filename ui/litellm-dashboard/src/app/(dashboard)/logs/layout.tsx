"use client";

import type { ReactNode } from "react";
import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AntDLoadingSpinner } from "@/components/ui/AntDLoadingSpinner";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { logsTabHref, slugFromPathname, type LogsTabSlug } from "@/app/(dashboard)/logs/tabRoutes";

const BASE_TAB_KEY = "request-logs";

const ORDERED_KEYS: Array<"" | LogsTabSlug> = ["", "audit", "deleted-keys", "deleted-teams"];

const TAB_LABELS: Record<"" | LogsTabSlug, string> = {
  "": "Request Logs",
  audit: "Audit Logs",
  "deleted-keys": "Deleted Keys",
  "deleted-teams": "Deleted Teams",
};

export default function LogsLayout({ children }: { children: ReactNode }) {
  const { accessToken, token, userRole, userId } = useAuthorized();
  const pathname = usePathname();
  const router = useRouter();

  const activeSlug = slugFromPathname(pathname);
  const isKnownSlug = ORDERED_KEYS.some((slug) => slug === activeSlug);
  const activeKey = isKnownSlug ? activeSlug || BASE_TAB_KEY : BASE_TAB_KEY;

  useEffect(() => {
    if (activeSlug !== "" && !isKnownSlug) {
      window.location.replace(logsTabHref(""));
    }
  }, [activeSlug, isKnownSlug]);

  const hasCredentials = Boolean(accessToken && token);
  const hasIdentity = Boolean(userRole && userId);

  if (!hasCredentials || !hasIdentity) {
    return (
      <div className="flex items-center justify-center h-64">
        <AntDLoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <div className="w-full p-6 overflow-x-hidden box-border">
      <Tabs value={activeKey} onValueChange={(key) => router.push(logsTabHref(key === BASE_TAB_KEY ? "" : key))}>
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
