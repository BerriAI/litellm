"use client";

import React from "react";
import { useTranslation } from "react-i18next";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import type { Team } from "@/components/key_team_helpers/key_list";
import { TabsContent, TabsTrigger } from "@/components/ui/tabs";
import { isProxyAdminRole } from "@/utils/roles";
import type { UsageOption } from "../UsageViewSelect/UsageViewSelect";
import TaskInsightsTab from "./TaskInsightsTab";

/** Task Insights covers the whole deployment, so it is offered on the global view to admin roles only. */
export const showsTaskInsights = (isAdmin: boolean, view: UsageOption): boolean => isAdmin && view === "global";

export function TaskInsightsTabTrigger({ visible }: { visible: boolean }) {
  const { t } = useTranslation();
  if (!visible) return null;
  return (
    <TabsTrigger value="tasks" className="flex-none px-3">
      {t("usage.pageView.tabs.taskInsights")}
    </TabsTrigger>
  );
}

interface TaskInsightsTabPanelProps {
  visible: boolean;
  active: boolean;
  startTime: Date | null;
  endTime: Date | null;
  teams: readonly Team[];
}

export function TaskInsightsTabPanel({ visible, active, startTime, endTime, teams }: TaskInsightsTabPanelProps) {
  const { userRole, isViewOnly } = useAuthorized();
  if (!visible) return null;
  return (
    <TabsContent value="tasks">
      <TaskInsightsTab
        startTime={startTime}
        endTime={endTime}
        teams={teams}
        canEdit={!isViewOnly && isProxyAdminRole(userRole ?? "")}
        active={active}
      />
    </TabsContent>
  );
}
