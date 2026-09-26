"use client";

import React, { useCallback, useEffect, useState } from "react";

import { getGeneralSettingsCall } from "@/components/networking";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { toast } from "@/lib/toast";
import {
  PromptCachingPanel,
  generalSettingsItem,
} from "@/app/(dashboard)/router-settings/_components/general_settings";
import CacheLeakageCard from "./CacheLeakageCard";
import PromptCachingRequestsTable from "./PromptCachingRequestsTable";
import { DailyActivityRange } from "./useDailyActivityRange";

interface PromptCachingTabProps {
  accessToken: string | null;
  activity: DailyActivityRange;
  scopeUserId: string | null;
}

const PromptCachingTab: React.FC<PromptCachingTabProps> = ({ accessToken, activity, scopeUserId }) => {
  const [settings, setSettings] = useState<generalSettingsItem[]>([]);

  const loadSettings = useCallback(() => {
    if (!accessToken) {
      return;
    }
    getGeneralSettingsCall(accessToken)
      .then((data: generalSettingsItem[]) => setSettings(data))
      .catch((error) => {
        console.error("Failed to load prompt caching settings:", error);
        toast.fromError("Failed to load prompt caching settings");
      });
  }, [accessToken]);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  const handleChange = (fieldName: string, newValue: unknown) => {
    setSettings((prev) =>
      prev.map((setting) => (setting.field_name === fieldName ? { ...setting, field_value: newValue } : setting)),
    );
  };

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full space-y-6">
      <PromptCachingPanel accessToken={accessToken} settings={settings} onChange={handleChange} />
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-muted-foreground">Date range for requests and cache leakage</p>
        <AdvancedDatePicker value={activity.dateValue} onValueChange={activity.onDateChange} />
      </div>
      <PromptCachingRequestsTable accessToken={accessToken} dateValue={activity.dateValue} />
      <CacheLeakageCard activity={activity} accessToken={accessToken} scopeUserId={scopeUserId} />
    </div>
  );
};

export default PromptCachingTab;
