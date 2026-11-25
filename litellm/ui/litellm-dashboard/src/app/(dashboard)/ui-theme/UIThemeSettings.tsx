import { PageContent } from "@/components/shared/Page";
import { PageHeader, PageHeaderTitle, PageHeaderDescription } from "@/components/shared/PageHeader";
import { Palette } from "lucide-react";
import React, { useState, useEffect } from "react";
import { useController } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { resolveThemeImages, useTheme } from "@/contexts/ThemeContext";
import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import { toast } from "@/lib/toast";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toUIThemeFields, toUIThemeUpdate } from "./mapper";
import { uiThemeSchema } from "./schema";

interface UIThemeSettingsProps {
  userID: string | null;
  userRole: string | null;
  accessToken: string | null;
}

const UIThemeSettings: React.FC<UIThemeSettingsProps> = ({ userID, userRole, accessToken }) => {
  const { t } = useTranslation();
  const { setLogoUrl, setLogoUrlDark, setFaviconUrl } = useTheme();
  const { control, handleSubmit, reset } = useZodForm(uiThemeSchema, { defaultValues: toUIThemeFields() });
  const { field: logoUrl } = useController({ control, name: "logoUrl" });
  const { field: logoUrlDark } = useController({ control, name: "logoUrlDark" });
  const { field: faviconUrl } = useController({ control, name: "faviconUrl" });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (accessToken) {
      fetchThemeSettings();
    }
  }, [accessToken]);

  const fetchThemeSettings = async () => {
    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl ? `${proxyBaseUrl}/get/ui_theme_settings` : "/get/ui_theme_settings";
      const response = await fetch(url, {
        method: "GET",
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      });
      if (response.ok) {
        const data = await response.json();
        const images = resolveThemeImages(data, proxyBaseUrl);
        reset(toUIThemeFields(data.values));
        setLogoUrl(images.logoUrl);
        setLogoUrlDark(images.logoUrlDark);
        setFaviconUrl(images.faviconUrl);
      }
    } catch (error) {
      console.error("Error fetching theme settings:", error);
    }
  };

  const handleSave = handleSubmit(async (values) => {
    const payload = toUIThemeUpdate(values);
    setLoading(true);
    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl ? `${proxyBaseUrl}/update/ui_theme_settings` : "/update/ui_theme_settings";
      const response = await fetch(url, {
        method: "PATCH",
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
      if (response.ok) {
        toast.success(t("uiTheme.settings.toasts.updated"));
        setLogoUrl(payload.logo_url);
        setLogoUrlDark(payload.logo_url_dark);
        setFaviconUrl(payload.favicon_url);
      } else {
        throw new Error("Failed to update settings");
      }
    } catch (error) {
      console.error("Error updating theme settings:", error);
      toast.fromError(t("uiTheme.settings.toasts.updateFailed"));
    } finally {
      setLoading(false);
    }
  });

  const handleReset = async () => {
    const defaults = toUIThemeFields();
    reset(defaults);
    setLogoUrl(null);
    setLogoUrlDark(null);
    setFaviconUrl(null);
    setLoading(true);
    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl ? `${proxyBaseUrl}/update/ui_theme_settings` : "/update/ui_theme_settings";
      const response = await fetch(url, {
        method: "PATCH",
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(toUIThemeUpdate(defaults)),
      });
      if (response.ok) {
        toast.success(t("uiTheme.settings.toasts.reset"));
      } else {
        throw new Error("Failed to reset");
      }
    } catch (error) {
      console.error("Error resetting theme settings:", error);
      toast.fromError(t("uiTheme.settings.toasts.resetFailed"));
    } finally {
      setLoading(false);
    }
  };

  if (!accessToken) {
    return null;
  }

  return (
    <PageContent>
      <PageHeader>
        <PageHeaderTitle>
          <Palette />
          {t("uiTheme.settings.title")}
        </PageHeaderTitle>
        <PageHeaderDescription>{t("uiTheme.settings.description")}</PageHeaderDescription>
      </PageHeader>
      <Card className="w-full max-w-4xl">
        <CardContent className="space-y-6">
          <div>
            <Label htmlFor="ui-theme-logo-url" className="mb-2">
              {t("uiTheme.settings.logoUrl.label")}
            </Label>
            <Input
              id="ui-theme-logo-url"
              placeholder="https://example.com/logo.png"
              value={logoUrl.value}
              onChange={(event) => {
                logoUrl.onChange(event);
                setLogoUrl(event.target.value || null);
              }}
            />
            <p className="mt-1 text-xs text-muted-foreground">{t("uiTheme.settings.logoUrl.hint")}</p>
          </div>
          <div>
            <Label htmlFor="ui-theme-logo-url-dark" className="mb-2">
              {t("uiTheme.settings.logoUrlDark.label")}
            </Label>
            <Input
              id="ui-theme-logo-url-dark"
              placeholder="https://example.com/logo-dark.png"
              value={logoUrlDark.value}
              onChange={(event) => {
                logoUrlDark.onChange(event);
                setLogoUrlDark(event.target.value || null);
              }}
            />
            <p className="mt-1 text-xs text-muted-foreground">{t("uiTheme.settings.logoUrlDark.hint")}</p>
          </div>
          <div>
            <Label htmlFor="ui-theme-favicon-url" className="mb-2">
              {t("uiTheme.settings.faviconUrl.label")}
            </Label>
            <Input
              id="ui-theme-favicon-url"
              placeholder="https://example.com/favicon.ico"
              value={faviconUrl.value}
              onChange={(event) => {
                faviconUrl.onChange(event);
                setFaviconUrl(event.target.value || null);
              }}
            />
            <p className="mt-1 text-xs text-muted-foreground">{t("uiTheme.settings.faviconUrl.hint")}</p>
          </div>
          <div className="flex gap-3 pt-4">
            <Button onClick={handleSave} disabled={loading}>
              {loading && <UiLoadingSpinner className="size-4" />}
              {t("uiTheme.settings.saveButton")}
            </Button>
            <Button variant="outline" onClick={handleReset} disabled={loading}>
              {loading && <UiLoadingSpinner className="size-4" />}
              {t("uiTheme.settings.resetButton")}
            </Button>
          </div>
        </CardContent>
      </Card>
    </PageContent>
  );
};

export default UIThemeSettings;
