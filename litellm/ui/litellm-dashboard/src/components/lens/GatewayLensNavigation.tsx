"use client";

import { createContext, useContext, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import {
  Activity,
  Bell,
  BookOpen,
  Bot,
  CalendarDays,
  ChartNoAxesColumn,
  Database,
  FlaskConical,
  House,
  Settings,
  Sparkles,
} from "lucide-react";
import { z } from "zod";
import { navMenuButtonClassName, navMenuClassName } from "@/components/shared/sidebarStyles";
import {
  Combobox,
  ComboboxContent,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
  ComboboxTrigger,
} from "@/components/ui/combobox";
import { SidebarMenu, SidebarMenuButton, SidebarMenuItem, useSidebar } from "@/components/ui/sidebar";
import { cn } from "@/lib/cva.config";

export const LENS_FRAME_CHANNEL = "litellm-lens:v1";

const option = z.object({ value: z.string(), label: z.string() });
const model = z.object({
  activeTab: z.string(),
  tabs: z.array(option.extend({ status: z.string().optional() })),
  currentAgent: z.string().nullable(),
  agents: z.array(option),
  agentLabel: z.string(),
  bookCallLabel: z.string(),
});

export const lensFrameMessage = z.discriminatedUnion("type", [
  z.object({ channel: z.literal(LENS_FRAME_CHANNEL), type: z.literal("ready") }),
  z.object({ channel: z.literal(LENS_FRAME_CHANNEL), type: z.literal("navigation"), navigation: model }),
]);

type Selection = { type: "tab" | "agent"; value: string };
type Navigation = {
  model: z.infer<typeof model>;
  select: (selection: Selection) => void;
};

const NavigationState = createContext<Navigation | null>(null);
const SetNavigation = createContext<(navigation: Navigation | null) => void>(() => {});

export function GatewayLensNavigationProvider({ children }: { children: ReactNode }) {
  const [navigation, setNavigation] = useState<Navigation | null>(null);
  return (
    <SetNavigation.Provider value={setNavigation}>
      <NavigationState.Provider value={navigation}>{children}</NavigationState.Provider>
    </SetNavigation.Provider>
  );
}

export function useSetLensNavigation() {
  return useContext(SetNavigation);
}

const tabIcons = new Map([
  ["home", House],
  ["agents", Bot],
  ["traces", Activity],
  ["alerts", Bell],
  ["profiles", ChartNoAxesColumn],
  ["findings", Sparkles],
  ["datasets", Database],
  ["evals", FlaskConical],
  ["optimizations", ChartNoAxesColumn],
  ["settings", Settings],
]);

export function GatewayLensNavigation({ collapsed }: { collapsed: boolean }) {
  const navigation = useContext(NavigationState);
  const { setOpenMobile } = useSidebar();
  const { t } = useTranslation();
  if (!navigation) return null;
  const { activeTab, tabs, agents, currentAgent, agentLabel, bookCallLabel } = navigation.model;
  const selectedAgent =
    currentAgent === null
      ? null
      : agents.find(({ value }) => value === currentAgent) ?? { value: currentAgent, label: currentAgent };
  const select = (selection: Selection) => {
    navigation.select(selection);
    setOpenMobile(false);
  };
  const menuClassName = collapsed ? navMenuClassName : undefined;
  const buttonClassName = collapsed ? navMenuButtonClassName : undefined;
  return (
    <nav aria-label={t("lensWorkspace.frameTitle")} className={cn("flex flex-col px-3 pb-3", menuClassName ?? "gap-2")}>
      <SidebarMenu className={menuClassName}>
        {!collapsed && selectedAgent !== null && (
          <SidebarMenuItem>
            <Combobox
              items={agents}
              value={selectedAgent}
              isItemEqualToValue={(item, selected) => item.value === selected.value}
              itemToStringLabel={(item) => item.label}
              onValueChange={(item) => {
                if (item) select({ type: "agent", value: item.value });
              }}
            >
              <ComboboxTrigger aria-label={`${agentLabel}: ${selectedAgent.label}`} render={<SidebarMenuButton />}>
                <Bot aria-hidden />
                <span className="min-w-0 flex-1 truncate">{selectedAgent.label}</span>
              </ComboboxTrigger>
              <ComboboxContent>
                <ComboboxInput aria-label={agentLabel} showTrigger={false} />
                <ComboboxList>
                  {(item: (typeof agents)[number]) => (
                    <ComboboxItem key={item.value} value={item}>
                      {item.label}
                    </ComboboxItem>
                  )}
                </ComboboxList>
              </ComboboxContent>
            </Combobox>
          </SidebarMenuItem>
        )}
        {tabs.map(({ value, label, status }) => {
          const Icon = tabIcons.get(value) ?? Activity;
          return (
            <SidebarMenuItem key={value}>
              <SidebarMenuButton
                isActive={value === activeTab}
                aria-current={value === activeTab ? "page" : undefined}
                aria-description={status}
                title={status}
                tooltip={label}
                className={buttonClassName}
                onClick={() => select({ type: "tab", value })}
              >
                <Icon aria-hidden />
                <span className={collapsed ? "sr-only" : undefined}>{label}</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          );
        })}
      </SidebarMenu>
      <SidebarMenu className={menuClassName}>
        <SidebarMenuItem>
          <SidebarMenuButton
            tooltip={t("lensWorkspace.docs")}
            className={buttonClassName}
            render={<a href="https://docs.litellm.ai/docs/proxy/lens" target="_blank" rel="noreferrer" />}
          >
            <BookOpen aria-hidden />
            <span className={collapsed ? "sr-only" : undefined}>{t("lensWorkspace.docs")}</span>
          </SidebarMenuButton>
        </SidebarMenuItem>
        <SidebarMenuItem>
          <SidebarMenuButton
            tooltip={bookCallLabel}
            className={buttonClassName}
            render={
              <a
                href="https://calendly.com/d/dtr8-nnr-3yy/litellm-product-team-feedback"
                target="_blank"
                rel="noreferrer"
              />
            }
          >
            <CalendarDays aria-hidden />
            <span className={collapsed ? "sr-only" : undefined}>{bookCallLabel}</span>
          </SidebarMenuButton>
        </SidebarMenuItem>
      </SidebarMenu>
    </nav>
  );
}
