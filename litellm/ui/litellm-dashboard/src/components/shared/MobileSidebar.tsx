"use client";

import { useEffect } from "react";
import { useSidebar } from "@/components/ui/sidebar";

export function useCloseMobileSidebarOn(navigationKey?: string) {
  const { isMobile, setOpenMobile } = useSidebar();
  useEffect(() => {
    setOpenMobile(false);
  }, [navigationKey, isMobile, setOpenMobile]);
}
