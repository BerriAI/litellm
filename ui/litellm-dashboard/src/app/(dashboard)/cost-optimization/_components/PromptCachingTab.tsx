"use client";

import { useLanguage } from "@/contexts/LanguageContext";
import React, { useEffect, useState } from "react";

import { getGeneralSettingsCall } from "@/components/networking";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { toast } from "@/lib/toast";
import TranslatedMessage from "@/components/shared/TranslatedMessage";
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
}

const PromptCachingTab: React.FC<PromptCachingTabProps> = ({ accessToken, activity }) => {
  const { t } = useLanguage();
  const [settings, setSettings] = useState<generalSettingsItem[]>([]);

  useEffect(() => {
    if (!accessToken) {
      return;
    }
    getGeneralSettingsCall(accessToken)
      .then((data: generalSettingsItem[]) => setSettings(data))
      .catch((error) => {
        // 只翻译固定通知，语言切换不重新请求或覆盖草稿。
        console.error("Failed to load prompt caching settings:", error);
        toast.fromError("Failed to load prompt caching settings", {
          description: <TranslatedMessage message="Failed to load prompt caching settings" />,
        });
      });
  }, [accessToken]);

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
        <p className="text-sm text-muted-foreground">{t("Date range for requests and cache leakage")}</p>
        <AdvancedDatePicker value={activity.dateValue} onValueChange={activity.onDateChange} />
      </div>
      <PromptCachingRequestsTable accessToken={accessToken} dateValue={activity.dateValue} />
      <CacheLeakageCard activity={activity} />
    </div>
  );
};

export default PromptCachingTab;
