"use client";

import { useEffect } from "react";
import { Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useSidebar } from "@/components/ui/sidebar";

export function MobileSidebarTrigger({ label }: { label: string }) {
  const { setOpenMobile } = useSidebar();
  return (
    <Button
      variant="ghost"
      size="icon"
      className="size-11 md:hidden"
      aria-label={label}
      onClick={() => setOpenMobile(true)}
    >
      <Menu />
    </Button>
  );
}

export function useCloseMobileSidebarOn(navigationKey?: string) {
  const { isMobile, setOpenMobile } = useSidebar();
  useEffect(() => {
    setOpenMobile(false);
  }, [navigationKey, isMobile, setOpenMobile]);
}
