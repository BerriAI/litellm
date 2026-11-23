"use client";

import type { ParseKeys } from "i18next";
import React, { useState } from "react";
import { useTranslation } from "react-i18next";
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

const SEVERITY_LABELS: Record<UserBannerSeverity, ParseKeys> = {
  info: "adminPanel.userBannerSettings.severity.options.info",
  warning: "adminPanel.userBannerSettings.severity.options.warning",
  error: "adminPanel.userBannerSettings.severity.options.error",
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
  const { t } = useTranslation();
  const [draft, setDraft] = useState<UserBannerUpdate>({
    enabled: persisted.enabled,
    message: persisted.message,
    severity: persisted.severity,
  });

  const messageMissing = draft.enabled && draft.message.trim() === "";

  const handleSave = () => {
    saveBanner(draft, {
      onSuccess: () => {
        toast.success(t("adminPanel.userBannerSettings.toasts.updated"));
      },
      onError: (error) => {
        toast.fromError(error);
      },
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("adminPanel.userBannerSettings.title")}</CardTitle>
        <CardDescription>{t("adminPanel.userBannerSettings.description")}</CardDescription>
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
                aria-label={t("adminPanel.userBannerSettings.publish")}
              />
              <Label>{t("adminPanel.userBannerSettings.publish")}</Label>
            </div>

            <div className="flex flex-col gap-2">
              <Label htmlFor="user-banner-message">{t("adminPanel.userBannerSettings.message.label")}</Label>
              <Textarea
                id="user-banner-message"
                value={draft.message}
                maxLength={4000}
                rows={3}
                placeholder={t("adminPanel.userBannerSettings.message.placeholder")}
                onChange={(event: React.ChangeEvent<HTMLTextAreaElement>) =>
                  setDraft({ ...draft, message: event.target.value })
                }
              />
              {messageMissing && (
                <p className="text-sm text-destructive">{t("adminPanel.userBannerSettings.message.missing")}</p>
              )}
            </div>

            <div className="flex flex-col gap-2">
              <Label>{t("adminPanel.userBannerSettings.severity.label")}</Label>
              <Select
                items={SEVERITY_ITEMS.map((item) => ({ value: item.value, label: t(item.label) }))}
                value={draft.severity}
                onValueChange={(value: string | null) =>
                  setDraft({ ...draft, severity: (value ?? "info") as UserBannerSeverity })
                }
              >
                <SelectTrigger className="w-48" aria-label={t("adminPanel.userBannerSettings.severity.ariaLabel")}>
                  <SelectValue placeholder={t("adminPanel.userBannerSettings.severity.placeholder")} />
                </SelectTrigger>
                <SelectContent>
                  {SEVERITY_ITEMS.map((item) => (
                    <SelectItem key={item.value} value={item.value}>
                      {t(item.label)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {draft.message.trim() !== "" && (
              <div className="flex flex-col gap-2">
                <Label>{t("adminPanel.userBannerSettings.preview")}</Label>
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
                {isPending ? t("adminPanel.userBannerSettings.saving") : t("adminPanel.userBannerSettings.save")}
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
