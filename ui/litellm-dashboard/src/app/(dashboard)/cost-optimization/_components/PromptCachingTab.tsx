"use client";

import React, { useCallback, useEffect, useState } from "react";

import { getGeneralSettingsCall } from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";
import {
  PromptCachingPanel,
  generalSettingsItem,
} from "@/app/(dashboard)/router-settings/_components/general_settings";
import CacheLeakageCard from "./CacheLeakageCard";
import { DailyActivityRange } from "./useDailyActivityRange";

interface PromptCachingTabProps {
  accessToken: string | null;
  activity: DailyActivityRange;
}

const PromptCachingTab: React.FC<PromptCachingTabProps> = ({ accessToken, activity }) => {
  const [settings, setSettings] = useState<generalSettingsItem[]>([]);

  const loadSettings = useCallback(() => {
    if (!accessToken) {
      return;
    }
    getGeneralSettingsCall(accessToken)
      .then((data: generalSettingsItem[]) => setSettings(data))
      .catch((error) => {
        console.error("Failed to load prompt caching settings:", error);
        NotificationsManager.fromBackend("Failed to load prompt caching settings");
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
      <CacheLeakageCard activity={activity} />
    </div>
  );
};

export default PromptCachingTab;
