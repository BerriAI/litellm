import type { ReactNode } from "react";
import { Layers } from "lucide-react";
import { SidebarMenuButton, SidebarMenuItem } from "@/components/ui/sidebar";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { controlPlanePages } from "@/lib/controlPlaneNavigation";

export const controlPlaneMenuGroup = {
  groupLabel: "controlPlane.globalControlPlane",
  items: Object.entries(controlPlanePages).map(([key, { label }]) => ({
    key,
    page: key,
    route: `control-plane/${key}`,
    label,
    icon: <Layers className="size-4 shrink-0" />,
  })),
} as const;

export function UnavailableControlPlaneItem({
  reason,
  className,
  children,
}: {
  reason: string;
  className: string;
  children: ReactNode;
}) {
  return (
    <SidebarMenuItem>
      <Tooltip>
        <TooltipTrigger render={<span tabIndex={0} className="block cursor-not-allowed" />}>
          <SidebarMenuButton aria-disabled className={className} disabled>
            {children}
          </SidebarMenuButton>
        </TooltipTrigger>
        <TooltipContent>{reason}</TooltipContent>
      </Tooltip>
    </SidebarMenuItem>
  );
}
