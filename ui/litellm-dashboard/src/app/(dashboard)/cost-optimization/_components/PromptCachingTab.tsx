"use client";

import React, { useCallback, useEffect, useState } from "react";

import { getGeneralSettingsCall } from "@/components/networking";
import { toast } from "@/lib/toast";
import {
  ENABLE_ANTHROPIC_PROMPT_CACHING,
  PromptCachingPanel,
  generalSettingsItem,
  isOn,
} from "@/app/(dashboard)/router-settings/_components/general_settings";
import CacheLeakageCard from "./CacheLeakageCard";
import PromptCachingTestCard from "./PromptCachingTestCard";
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

  const enableSetting = settings.find((s) => s.field_name === ENABLE_ANTHROPIC_PROMPT_CACHING);

  return (
    <div className="w-full space-y-6">
      <PromptCachingPanel accessToken={accessToken} settings={settings} onChange={handleChange} />
      <PromptCachingTestCard accessToken={accessToken} enabled={isOn(enableSetting?.field_value)} />
      <CacheLeakageCard activity={activity} />
    </div>
  );
};

export default PromptCachingTab;
