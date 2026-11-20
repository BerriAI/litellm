"use client";

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import type { LensHost } from "@litellm/lens-ui";
import { useSidebar } from "@/components/ui/sidebar";

type LensNavigation = NonNullable<LensHost["navigation"]>;

interface GatewayLensNavigation {
  readonly target: HTMLDivElement | null;
  readonly setTarget: (target: HTMLDivElement | null) => void;
  readonly navigation: LensNavigation;
}

const GatewayLensNavigationContext = createContext<GatewayLensNavigation | null>(null);

function NavigationContainer({ children }: { readonly children: ReactNode }) {
  const context = useContext(GatewayLensNavigationContext);
  return context?.target ? createPortal(children, context.target) : null;
}

export function GatewayLensNavigationProvider({ children }: { readonly children: ReactNode }) {
  const [target, setTarget] = useState<HTMLDivElement | null>(null);
  const { state, isMobile, setOpenMobile } = useSidebar();
  const onNavigate = useCallback(() => setOpenMobile(false), [setOpenMobile]);
  const navigation = useMemo<LensNavigation>(
    () => ({ Container: NavigationContainer, collapsed: state === "collapsed" && !isMobile, onNavigate }),
    [state, isMobile, onNavigate],
  );
  const context = useMemo(() => ({ target, setTarget, navigation }), [target, navigation]);
  return <GatewayLensNavigationContext.Provider value={context}>{children}</GatewayLensNavigationContext.Provider>;
}

export function GatewayLensNavigationTarget() {
  const context = useContext(GatewayLensNavigationContext);
  return <div ref={context?.setTarget} className="lens-workspace lens-shell flex min-h-0 flex-1 flex-col gap-1" />;
}

export function useGatewayLensNavigation(): LensHost["navigation"] {
  return useContext(GatewayLensNavigationContext)?.navigation;
}
