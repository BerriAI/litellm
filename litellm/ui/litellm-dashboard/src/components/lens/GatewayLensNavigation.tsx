"use client";

import { LENS_NAVIGATION_ITEMS, useLensNavigation } from "@litellm/lens-ui";
import { navMenuButtonClassName, navMenuClassName } from "@/components/shared/sidebarStyles";
import { SidebarMenu, SidebarMenuButton, SidebarMenuItem, useSidebar } from "@/components/ui/sidebar";

export function GatewayLensNavigation() {
  const { tab, navigate } = useLensNavigation();
  const { state, isMobile, setOpenMobile } = useSidebar();
  const collapsed = state === "collapsed" && !isMobile;

  return (
    <SidebarMenu className={navMenuClassName}>
      {LENS_NAVIGATION_ITEMS.map(({ value, label, icon: Icon }) => (
        <SidebarMenuItem key={value}>
          <SidebarMenuButton
            isActive={tab === value}
            aria-current={tab === value ? "page" : undefined}
            aria-label={label}
            title={collapsed ? label : undefined}
            tooltip={label}
            className={navMenuButtonClassName}
            onClick={() => {
              navigate(value);
              if (isMobile) setOpenMobile(false);
            }}
          >
            <Icon aria-hidden="true" strokeWidth={1.75} />
            <span className="flex-1 truncate group-data-[collapsible=icon]:hidden">{label}</span>
          </SidebarMenuButton>
        </SidebarMenuItem>
      ))}
    </SidebarMenu>
  );
}
