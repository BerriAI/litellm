"use client";

import { type ReactNode, type UIEvent, useCallback, useEffect, useRef, useSyncExternalStore } from "react";
import { Dialog as DialogPrimitive } from "@base-ui/react/dialog";
import { Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Sidebar, useSidebar } from "@/components/ui/sidebar";

export type MobileSidebarHandle = DialogPrimitive.Handle<unknown>;

export const createMobileSidebarHandle = (): MobileSidebarHandle => DialogPrimitive.createHandle();

export function MobileSidebarTrigger({ label, handle }: { label: string; handle: MobileSidebarHandle }) {
  return (
    <SheetTrigger
      handle={handle}
      render={<Button variant="ghost" size="icon" className="size-11 md:hidden" aria-label={label} />}
    >
      <Menu />
    </SheetTrigger>
  );
}

const DESKTOP_QUERY = "(min-width: 768px)";

const subscribeToDesktopQuery = (onChange: () => void) => {
  const media = window.matchMedia(DESKTOP_QUERY);
  media.addEventListener("change", onChange);
  return () => media.removeEventListener("change", onChange);
};

const useIsDesktop = (): boolean =>
  useSyncExternalStore(
    subscribeToDesktopQuery,
    () => window.matchMedia(DESKTOP_QUERY).matches,
    () => false,
  );

export function useShellSidebar() {
  const sidebar = useSidebar();
  const isDesktop = useIsDesktop();
  const drawerOpen = !isDesktop && sidebar.openMobile;
  const inDrawer = sidebar.isMobile || drawerOpen;
  return { ...sidebar, isDesktop, drawerOpen, inDrawer, collapsed: sidebar.state === "collapsed" && !inDrawer };
}

export function useCloseMobileSidebarOn(navigationKey?: string) {
  const { isMobile, isDesktop, setOpenMobile } = useShellSidebar();
  useEffect(() => {
    setOpenMobile(false);
  }, [navigationKey, isMobile, isDesktop, setOpenMobile]);
}

const SCROLL_VIEWPORT_SLOT = "scroll-area-viewport";

function useScrollMemory() {
  const scrollTop = useRef(0);
  const remember = useCallback((event: UIEvent<HTMLDivElement>) => {
    if (event.target instanceof HTMLElement && event.target.dataset.slot === SCROLL_VIEWPORT_SLOT) {
      scrollTop.current = event.target.scrollTop;
    }
  }, []);
  const restore = useCallback((node: HTMLDivElement | null) => {
    const viewport = node?.querySelector<HTMLElement>(`[data-slot="${SCROLL_VIEWPORT_SLOT}"]`);
    if (viewport) viewport.scrollTop = scrollTop.current;
  }, []);
  return { ref: restore, onScrollCapture: remember };
}

interface ShellSidebarProps {
  handle: MobileSidebarHandle;
  label?: string;
  sheetTitle: string;
  sheetDescription?: string;
  sheetInitialFocus?: false;
  keepScrollAcrossBreakpoint?: boolean;
  children: ReactNode;
}

export function ShellSidebar({
  handle,
  label,
  sheetTitle,
  sheetDescription,
  sheetInitialFocus,
  keepScrollAcrossBreakpoint = false,
  children,
}: ShellSidebarProps) {
  const { isMobile, drawerOpen, setOpenMobile } = useShellSidebar();
  const scrollMemory = useScrollMemory();
  return (
    <>
      {!isMobile && (
        <Sidebar
          {...(keepScrollAcrossBreakpoint ? scrollMemory : undefined)}
          collapsible="icon"
          role="complementary"
          aria-label={label}
          className="border-sidebar-border ease-in-out"
        >
          {children}
        </Sidebar>
      )}
      <Sheet handle={handle} open={drawerOpen} onOpenChange={setOpenMobile}>
        <SheetContent
          side="left"
          initialFocus={sheetInitialFocus}
          showCloseButton={false}
          className="gap-0 p-0 data-[side=left]:w-[min(280px,calc(100vw-3rem))]"
        >
          <SheetHeader className="sr-only">
            <SheetTitle>{sheetTitle}</SheetTitle>
            {sheetDescription && <SheetDescription>{sheetDescription}</SheetDescription>}
          </SheetHeader>
          <Sidebar
            collapsible="none"
            role="complementary"
            className="w-full flex-none overflow-hidden border-r border-sidebar-border"
          >
            {children}
          </Sidebar>
        </SheetContent>
      </Sheet>
    </>
  );
}
