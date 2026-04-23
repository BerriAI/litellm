import React, { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useTheme } from "@/contexts/ThemeContext";
import {
  getProxyBaseUrl,
  getGlobalLitellmHeaderName,
} from "@/components/networking";
import NotificationsManager from "./molecules/notifications_manager";

interface UIThemeSettingsProps {
  userID: string | null;
  userRole: string | null;
  accessToken: string | null;
}

const UIThemeSettings: React.FC<UIThemeSettingsProps> = ({
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  userID,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  userRole,
  accessToken,
}) => {
  const { setLogoUrl, setFaviconUrl } = useTheme();
  const [logoUrlInput, setLogoUrlInput] = useState<string>("");
  const [faviconUrlInput, setFaviconUrlInput] = useState<string>("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (accessToken) fetchThemeSettings();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken]);

  const fetchThemeSettings = async () => {
    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl
        ? `${proxyBaseUrl}/get/ui_theme_settings`
        : "/get/ui_theme_settings";
      const response = await fetch(url, {
        method: "GET",
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      });
      if (response.ok) {
        const data = await response.json();
        setLogoUrlInput(data.values?.logo_url || "");
        setFaviconUrlInput(data.values?.favicon_url || "");
        setLogoUrl(data.values?.logo_url || null);
        setFaviconUrl(data.values?.favicon_url || null);
      }
    } catch (error) {
      console.error("Error fetching theme settings:", error);
    }
  };

  const handleSave = async () => {
    setLoading(true);
    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl
        ? `${proxyBaseUrl}/update/ui_theme_settings`
        : "/update/ui_theme_settings";
      const response = await fetch(url, {
        method: "PATCH",
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          logo_url: logoUrlInput || null,
          favicon_url: faviconUrlInput || null,
        }),
      });
      if (response.ok) {
        NotificationsManager.success("Theme settings updated successfully!");
        setLogoUrl(logoUrlInput || null);
        setFaviconUrl(faviconUrlInput || null);
      } else {
        throw new Error("Failed to update settings");
      }
    } catch (error) {
      console.error("Error updating theme settings:", error);
      NotificationsManager.fromBackend("Failed to update theme settings");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    setLogoUrlInput("");
    setFaviconUrlInput("");
    setLogoUrl(null);
    setFaviconUrl(null);
    setLoading(true);
    try {
      const proxyBaseUrl = getProxyBaseUrl();
      const url = proxyBaseUrl
        ? `${proxyBaseUrl}/update/ui_theme_settings`
        : "/update/ui_theme_settings";
      const response = await fetch(url, {
        method: "PATCH",
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ logo_url: null, favicon_url: null }),
      });
      if (response.ok) {
        NotificationsManager.success("Theme settings reset to default!");
      } else {
        throw new Error("Failed to reset");
      }
    } catch (error) {
      console.error("Error resetting theme settings:", error);
      NotificationsManager.fromBackend("Failed to reset theme settings");
    } finally {
      setLoading(false);
    }
  };

  if (!accessToken) return null;

  return (
    <div className="w-full mx-auto max-w-4xl px-6 py-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-2">UI Theme Customization</h1>
        <p className="text-muted-foreground">
          Customize your LiteLLM admin dashboard with a custom logo and favicon.
        </p>
      </div>
      <Card className="shadow-sm p-6">
        <div className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="ui-theme-logo">Custom Logo URL</Label>
            <Input
              id="ui-theme-logo"
              placeholder="https://example.com/logo.png"
              value={logoUrlInput}
              onChange={(e) => {
                const v = e.target.value;
                setLogoUrlInput(v);
                setLogoUrl(v || null);
              }}
            />
            <p className="text-xs text-muted-foreground">
              Enter a URL for your custom logo or leave empty for default
            </p>
          </div>
          <div className="space-y-2">
            <Label htmlFor="ui-theme-favicon">Custom Favicon URL</Label>
            <Input
              id="ui-theme-favicon"
              placeholder="https://example.com/favicon.ico"
              value={faviconUrlInput}
              onChange={(e) => {
                const v = e.target.value;
                setFaviconUrlInput(v);
                setFaviconUrl(v || null);
              }}
            />
            <p className="text-xs text-muted-foreground">
              Enter a URL for your custom favicon (.ico, .png, or .svg) or leave
              empty for default
            </p>
          </div>
          <div className="flex gap-3 pt-4">
            <Button onClick={handleSave} disabled={loading}>
              {loading ? "Saving…" : "Save Changes"}
            </Button>
            <Button
              variant="secondary"
              onClick={handleReset}
              disabled={loading}
            >
              Reset to Default
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default UIThemeSettings;
