"use client";

import { type ReactNode, useEffect } from "react";
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

export function useCloseMobileSidebarOn(navigationKey?: string) {
  const { isMobile, setOpenMobile } = useSidebar();
  useEffect(() => {
    setOpenMobile(false);
  }, [navigationKey, isMobile, setOpenMobile]);
}

interface ShellSidebarProps {
  handle: MobileSidebarHandle;
  label?: string;
  sheetTitle: string;
  sheetDescription?: string;
  sheetInitialFocus?: false;
  children: ReactNode;
}

export function ShellSidebar({
  handle,
  label,
  sheetTitle,
  sheetDescription,
  sheetInitialFocus,
  children,
}: ShellSidebarProps) {
  const { isMobile, openMobile, setOpenMobile } = useSidebar();
  return (
    <>
      {!isMobile && (
        <Sidebar
          collapsible="icon"
          role="complementary"
          aria-label={label}
          className="border-sidebar-border ease-in-out"
        >
          {children}
        </Sidebar>
      )}
      <Sheet handle={handle} open={isMobile && openMobile} onOpenChange={setOpenMobile}>
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
