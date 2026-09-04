import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import type { TabRoutes } from "@/utils/tabRoutes";

interface UseTabRoutingArgs {
  routes: Pick<TabRoutes<string>, "tabHref" | "slugFromPathname">;
  baseTabKey: string;
  visibleKeys: readonly string[];
  ready?: boolean;
}

interface TabRoutingState {
  activeSlug: string;
  activeKey: string;
  onTabChange: (key: string) => void;
}

export function useTabRouting({ routes, baseTabKey, visibleKeys, ready = true }: UseTabRoutingArgs): TabRoutingState {
  const { tabHref, slugFromPathname } = routes;
  const pathname = usePathname();
  const router = useRouter();

  const activeSlug = slugFromPathname(pathname);
  const isKnownSlug = activeSlug === "" || visibleKeys.includes(activeSlug);
  const activeKey = isKnownSlug ? activeSlug || baseTabKey : baseTabKey;

  useEffect(() => {
    if (ready && activeSlug !== "" && !isKnownSlug) {
      window.location.replace(tabHref(""));
    }
  }, [ready, activeSlug, isKnownSlug, tabHref]);

  const onTabChange = (key: string) => {
    router.push(tabHref(key === baseTabKey ? "" : key));
  };

  return { activeSlug, activeKey, onTabChange };
}
