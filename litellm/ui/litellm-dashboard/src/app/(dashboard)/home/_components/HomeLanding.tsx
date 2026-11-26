"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { useVisibleMenuGroups } from "@/components/CommandPalette/useVisibleMenuGroups";
import { flattenNavItems } from "@/components/CommandPalette/utils";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import type { MenuGroup } from "@/components/LeftNav";
import { uiHref } from "@/utils/uiHref";
import HomePage from "./HomePage";

export const routeReplacingHiddenHome = (groups: readonly MenuGroup[]): string | null => {
  const routes = flattenNavItems(groups).map(({ route }) => route);
  return routes.includes("home") ? null : routes[0] ?? null;
};

export default function HomeLanding() {
  const router = useRouter();
  const { isLoading: isAuthLoading } = useAuthorized();
  const { isLoading: isSettingsLoading } = useUISettings();
  const groups = useVisibleMenuGroups();
  const isLoading = isAuthLoading || isSettingsLoading;
  const redirectRoute = isLoading ? null : routeReplacingHiddenHome(groups);

  useEffect(() => {
    if (redirectRoute !== null) router.replace(uiHref(redirectRoute));
  }, [redirectRoute, router]);

  if (isLoading || redirectRoute !== null) return <LoadingScreen />;
  return <HomePage />;
}
