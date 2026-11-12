import type { CSSProperties } from "react";

const navButtonLook = [
  "relative h-[34px] gap-2.5 px-2.5 py-0 text-[13px] font-medium text-sidebar-foreground/70 no-underline transition-colors",
  "max-md:min-h-11 [&_svg]:size-[18px]",
  "data-active:before:absolute data-active:before:inset-y-1.5 data-active:before:left-0 data-active:before:w-[3px] data-active:before:rounded-r-full data-active:before:bg-sidebar-primary",
].join(" ");

export const navMenuButtonClassName = [
  navButtonLook,
  "group-data-[collapsible=icon]:mx-auto group-data-[collapsible=icon]:size-9! group-data-[collapsible=icon]:gap-0 group-data-[collapsible=icon]:p-[9px]! group-data-[collapsible=icon]:before:hidden",
].join(" ");

export const navMenuSubButtonClassName = [
  navButtonLook,
  "w-full translate-x-0 data-[size=md]:text-[13px] [&>svg]:size-[18px] [&>svg]:text-current",
].join(" ");

export const navMenuSubClassName = "my-0.5 translate-x-0 gap-0.5 px-0 pl-3";

export const navMenuClassName = "gap-0.5";

export const sidebarProviderClassName = "[&_[data-slot=sidebar-gap]]:ease-in-out";

export const sidebarWidthStyle = { "--sidebar-width": "280px", "--sidebar-width-icon": "72px" } as CSSProperties;
