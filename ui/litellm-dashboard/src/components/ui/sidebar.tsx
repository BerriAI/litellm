"use client";

import * as React from "react";
import { Button as ButtonPrimitive } from "@base-ui/react/button";
import { type VariantProps } from "cva";

import { cn, cva } from "@/lib/cva.config";

type SidebarContextValue = { collapsed: boolean };
const SidebarContext = React.createContext<SidebarContextValue>({ collapsed: false });

export function useSidebar(): SidebarContextValue {
  return React.useContext(SidebarContext);
}

const Sidebar = React.forwardRef<HTMLElement, React.ComponentPropsWithoutRef<"aside"> & { collapsed?: boolean }>(
  ({ className, collapsed = false, children, ...props }, ref) => (
    <SidebarContext.Provider value={{ collapsed }}>
      <aside
        ref={ref}
        data-slot="sidebar"
        data-collapsed={collapsed}
        className={cn(
          "group/sidebar flex h-full flex-none flex-col overflow-hidden border-r border-sidebar-border bg-sidebar text-sidebar-foreground transition-[width] duration-200 ease-in-out",
          collapsed ? "w-[72px]" : "w-[280px]",
          className,
        )}
        {...props}
      >
        {children}
      </aside>
    </SidebarContext.Provider>
  ),
);
Sidebar.displayName = "Sidebar";

const SidebarHeader = React.forwardRef<HTMLDivElement, React.ComponentPropsWithoutRef<"div">>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      data-slot="sidebar-header"
      className={cn("flex flex-none flex-col gap-2 p-3", className)}
      {...props}
    />
  ),
);
SidebarHeader.displayName = "SidebarHeader";

const SidebarContent = React.forwardRef<HTMLElement, React.ComponentPropsWithoutRef<"nav">>(
  ({ className, ...props }, ref) => (
    <nav
      ref={ref}
      data-slot="sidebar-content"
      className={cn("flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto px-3 pb-3", className)}
      {...props}
    />
  ),
);
SidebarContent.displayName = "SidebarContent";

const SidebarFooter = React.forwardRef<HTMLDivElement, React.ComponentPropsWithoutRef<"div">>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      data-slot="sidebar-footer"
      className={cn("flex flex-none flex-col gap-2.5 border-t border-sidebar-border p-3", className)}
      {...props}
    />
  ),
);
SidebarFooter.displayName = "SidebarFooter";

const SidebarGroup = React.forwardRef<HTMLDivElement, React.ComponentPropsWithoutRef<"div">>(
  ({ className, ...props }, ref) => (
    <div ref={ref} data-slot="sidebar-group" className={cn("flex flex-col gap-0.5 py-1", className)} {...props} />
  ),
);
SidebarGroup.displayName = "SidebarGroup";

const SidebarGroupLabel = React.forwardRef<HTMLDivElement, React.ComponentPropsWithoutRef<"div">>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      data-slot="sidebar-group-label"
      className={cn(
        "px-2 pt-3 pb-1.5 text-[11px] font-semibold tracking-wider text-muted-foreground uppercase group-data-[collapsed=true]/sidebar:hidden",
        className,
      )}
      {...props}
    />
  ),
);
SidebarGroupLabel.displayName = "SidebarGroupLabel";

const SidebarMenu = React.forwardRef<HTMLUListElement, React.ComponentPropsWithoutRef<"ul">>(
  ({ className, ...props }, ref) => (
    <ul ref={ref} data-slot="sidebar-menu" className={cn("flex w-full flex-col gap-0.5", className)} {...props} />
  ),
);
SidebarMenu.displayName = "SidebarMenu";

const SidebarMenuItem = React.forwardRef<HTMLLIElement, React.ComponentPropsWithoutRef<"li">>(
  ({ className, ...props }, ref) => (
    <li ref={ref} data-slot="sidebar-menu-item" className={cn("relative", className)} {...props} />
  ),
);
SidebarMenuItem.displayName = "SidebarMenuItem";

const SidebarMenuSub = React.forwardRef<HTMLUListElement, React.ComponentPropsWithoutRef<"ul">>(
  ({ className, ...props }, ref) => (
    <ul
      ref={ref}
      data-slot="sidebar-menu-sub"
      className={cn(
        "mx-3.5 my-0.5 flex min-w-0 flex-col gap-0.5 border-l border-sidebar-border py-0.5 pl-3 group-data-[collapsed=true]/sidebar:hidden",
        className,
      )}
      {...props}
    />
  ),
);
SidebarMenuSub.displayName = "SidebarMenuSub";

const SidebarMenuBadge = React.forwardRef<HTMLSpanElement, React.ComponentPropsWithoutRef<"span">>(
  ({ className, ...props }, ref) => (
    <span
      ref={ref}
      data-slot="sidebar-menu-badge"
      className={cn(
        "ml-auto flex-none rounded-full bg-sidebar-primary/10 px-1.5 py-px text-[10px] font-semibold text-sidebar-primary tabular-nums group-data-[collapsed=true]/sidebar:hidden",
        className,
      )}
      {...props}
    />
  ),
);
SidebarMenuBadge.displayName = "SidebarMenuBadge";

const sidebarMenuButtonVariants = cva({
  base: [
    "group/menu-btn relative flex w-full items-center gap-2.5 overflow-hidden rounded-md px-2.5 text-left text-[13px] font-medium no-underline",
    "text-sidebar-foreground/70 outline-none transition-colors",
    "hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
    "focus-visible:ring-2 focus-visible:ring-sidebar-ring",
    "disabled:pointer-events-none disabled:opacity-50",
    "[&>svg]:size-[18px] [&>svg]:shrink-0",
    "group-data-[collapsed=true]/sidebar:mx-auto group-data-[collapsed=true]/sidebar:size-9 group-data-[collapsed=true]/sidebar:justify-center group-data-[collapsed=true]/sidebar:gap-0 group-data-[collapsed=true]/sidebar:px-0",
  ].join(" "),
  variants: {
    isActive: {
      true: "bg-sidebar-accent text-sidebar-accent-foreground before:absolute before:inset-y-1.5 before:left-0 before:w-[3px] before:rounded-r-full before:bg-sidebar-primary group-data-[collapsed=true]/sidebar:before:hidden",
      false: "",
    },
    size: {
      default: "h-[34px]",
      sub: "h-[34px]",
    },
  },
  defaultVariants: { isActive: false, size: "default" },
});

type SidebarMenuButtonProps = ButtonPrimitive.Props & VariantProps<typeof sidebarMenuButtonVariants>;

const SidebarMenuButton = React.forwardRef<HTMLButtonElement, SidebarMenuButtonProps>(
  ({ className, isActive, size, ...props }, ref) => (
    <ButtonPrimitive
      ref={ref}
      data-slot="sidebar-menu-button"
      data-active={isActive || undefined}
      className={cn(sidebarMenuButtonVariants({ isActive, size, className }))}
      {...props}
    />
  ),
);
SidebarMenuButton.displayName = "SidebarMenuButton";

const SidebarSeparator = React.forwardRef<HTMLDivElement, React.ComponentPropsWithoutRef<"div">>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      data-slot="sidebar-separator"
      className={cn("mx-2 my-2 h-px bg-sidebar-border", className)}
      {...props}
    />
  ),
);
SidebarSeparator.displayName = "SidebarSeparator";

export {
  Sidebar,
  SidebarHeader,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarMenuSub,
  SidebarMenuBadge,
  SidebarSeparator,
  sidebarMenuButtonVariants,
};
