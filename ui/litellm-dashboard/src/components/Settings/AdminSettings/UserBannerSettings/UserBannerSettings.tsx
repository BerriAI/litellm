"use client";

import React, { useState } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useUpdateUserBanner } from "@/app/(dashboard)/hooks/userBanner/useUpdateUserBanner";
import { useUserBanner } from "@/app/(dashboard)/hooks/userBanner/useUserBanner";
import { toast } from "@/lib/toast";
import { UserBanner, UserBannerSeverity, UserBannerUpdate } from "@/components/networking";
import { Alert, AlertDescription } from "@/components/shared/Alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { SEVERITY_ICONS, UserBannerMarkdown } from "@/components/UserBanner";
import { Skeleton } from "@/components/ui/skeleton";
import { useLanguage } from "@/contexts/LanguageContext";

const SEVERITY_LABELS: Record<UserBannerSeverity, string> = {
  info: "Info",
  warning: "Warning",
  error: "Error",
};

const SEVERITY_ITEMS = (Object.keys(SEVERITY_LABELS) as UserBannerSeverity[]).map((severity) => ({
  value: severity,
  label: SEVERITY_LABELS[severity],
}));

const EMPTY_BANNER: UserBanner = { enabled: false, message: "", severity: "info", revision: "" };

export default function UserBannerSettings() {
  const { accessToken } = useAuthorized();
  const { data: banner, isLoading } = useUserBanner(accessToken);
  const { mutate: saveBanner, isPending } = useUpdateUserBanner(accessToken);
  const persisted = banner ?? EMPTY_BANNER;

  return (
    <UserBannerSettingsForm
      key={JSON.stringify(persisted)}
      persisted={persisted}
      isLoading={isLoading}
      isPending={isPending}
      saveBanner={saveBanner}
    />
  );
}

interface UserBannerSettingsFormProps {
  persisted: UserBanner;
  isLoading: boolean;
  isPending: boolean;
  saveBanner: ReturnType<typeof useUpdateUserBanner>["mutate"];
}

function UserBannerSettingsForm({ persisted, isLoading, isPending, saveBanner }: UserBannerSettingsFormProps) {
  const { t } = useLanguage();
  // 仅翻译控件文案，管理员撰写的公告内容始终保留原文
  const severityItems = SEVERITY_ITEMS.map((item) => ({ ...item, label: t(item.label) }));
  const [draft, setDraft] = useState<UserBannerUpdate>({
    enabled: persisted.enabled,
    message: persisted.message,
    severity: persisted.severity,
  });

  const messageMissing = draft.enabled && draft.message.trim() === "";

  const handleSave = () => {
    saveBanner(draft, {
      onSuccess: () => {
        toast.success(t("User banner updated successfully"));
      },
      onError: (error) => {
        toast.fromError(error);
      },
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("User Banner")}</CardTitle>
        <CardDescription>
          {t(
            "Publish an announcement to all dashboard users. Markdown is supported; the banner appears below the header on every page until you unpublish it. Users can dismiss it, and it reappears whenever the content changes.",
          )}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-40 w-full" />
        ) : (
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-3">
              <Switch
                checked={draft.enabled}
                onCheckedChange={(checked: boolean) => setDraft({ ...draft, enabled: checked })}
                aria-label={t("Publish user banner")}
              />
              <Label>{t("Publish user banner")}</Label>
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="user-banner-message">{t("Message")}</Label>
              <Textarea
                id="user-banner-message"
                value={draft.message}
                maxLength={4000}
                rows={3}
                placeholder={t(
                  "**Scheduled maintenance** tonight at 10 PM UTC. See [status page](https://example.com).",
                )}
                onChange={(event: React.ChangeEvent<HTMLTextAreaElement>) =>
                  setDraft({ ...draft, message: event.target.value })
                }
              />
              {messageMissing && <p className="text-sm text-destructive">{t("Add a message before publishing.")}</p>}
            </div>

            <div className="flex flex-col gap-2">
              <Label>{t("Severity")}</Label>
              <Select
                items={severityItems}
                value={draft.severity}
                onValueChange={(value: string | null) =>
                  setDraft({ ...draft, severity: (value ?? "info") as UserBannerSeverity })
                }
              >
                <SelectTrigger className="w-48" aria-label={t("Banner severity")}>
                  <SelectValue placeholder={t("Severity")} />
                </SelectTrigger>
                <SelectContent>
                  {severityItems.map((item) => (
                    <SelectItem key={item.value} value={item.value}>
                      {item.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {draft.message.trim() !== "" && (
              <div className="flex flex-col gap-2">
                <Label>{t("Preview")}</Label>
                <Alert variant={draft.severity}>
                  {SEVERITY_ICONS[draft.severity]}
                  <AlertDescription>
                    <UserBannerMarkdown message={draft.message} />
                  </AlertDescription>
                </Alert>
              </div>
            )}

            <div className="flex justify-end">
              <Button onClick={handleSave} disabled={isPending || messageMissing}>
                {t(isPending ? "Saving..." : "Save banner")}
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
