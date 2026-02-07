import React, { useState, useEffect } from "react";
import { Card, Title, Text, TextInput, Button } from "@tremor/react";
import { useTheme } from "@/contexts/ThemeContext";
import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import NotificationsManager from "./molecules/notifications_manager";

interface UIThemeSettingsProps {
  userID: string | null;
  userRole: string | null;
  accessToken: string | null;
}

const UIThemeSettings: React.FC<UIThemeSettingsProps> = ({ userID, userRole, accessToken }) => {
  const { logoUrl, setLogoUrl } = useTheme();
  const [logoUrlInput, setLogoUrlInput] = useState<string>("");
  const [loading, setLoading] = useState(false);

  // Load current settings when component mounts
  useEffect(() => {
    if (accessToken) {
      fetchLogoSettings();
    }
  }, [accessToken]);

  const fetchLogoSettings = async () => {
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
        const logoUrl = data.values?.logo_url || "";
        setLogoUrlInput(logoUrl);
        setLogoUrl(logoUrl || null);
      }
    } catch (error) {
      console.error("Error fetching theme settings:", error);
    }
  };

  const handleSave = async () => {
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
        body: JSON.stringify({
          logo_url: logoUrlInput || null,
        }),
      });

      if (response.ok) {
        NotificationsManager.success("Logo settings updated successfully!");
        setLogoUrl(logoUrlInput || null);
      } else {
        throw new Error("Failed to update settings");
      }
    } catch (error) {
      console.error("Error updating logo settings:", error);
      NotificationsManager.fromBackend("Failed to update logo settings");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    setLogoUrlInput("");
    setLogoUrl(null);

    // Save null to backend to clear the logo
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
        body: JSON.stringify({
          logo_url: null,
        }),
      });

      if (response.ok) {
        NotificationsManager.success("Logo reset to default!");
      } else {
        throw new Error("Failed to reset logo");
      }
    } catch (error) {
      console.error("Error resetting logo:", error);
      NotificationsManager.fromBackend("Failed to reset logo");
    } finally {
      setLoading(false);
    }
  };

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full mx-auto max-w-4xl px-6 py-8">
      <div className="mb-8">
        <Title className="text-2xl font-bold mb-2">Logo Customization</Title>
        <Text className="text-gray-600">Customize your LiteLLM admin dashboard with a custom logo.</Text>
      </div>

      <Card className="shadow-sm p-6">
        <div className="space-y-6">
          <div>
            <Text className="text-sm font-medium text-gray-700 mb-2 block">Custom Logo URL</Text>
            <TextInput
              placeholder="https://example.com/logo.png"
              value={logoUrlInput}
              onValueChange={(value) => {
                setLogoUrlInput(value);
                // Update logo in real-time for preview
                setLogoUrl(value || null);
              }}
              className="w-full"
            />
            <Text className="text-xs text-gray-500 mt-1">
              Enter a URL for your custom logo or leave empty to use the default LiteLLM logo
            </Text>
          </div>

          {/* Logo Preview */}
          <div>
            <Text className="text-sm font-medium text-gray-700 mb-2 block">Current Logo</Text>
            <div className="bg-gray-50 rounded-lg p-6 flex items-center justify-center min-h-[120px]">
              {logoUrlInput ? (
                <img
                  src={logoUrlInput}
                  alt="Custom logo"
                  className="max-w-full max-h-24 object-contain"
                  onError={(e) => {
                    const target = e.target as HTMLImageElement;
                    target.style.display = "none";
                    const fallbackText = document.createElement("div");
                    fallbackText.className = "text-gray-500 text-sm";
                    fallbackText.textContent = "Failed to load image";
                    target.parentElement?.appendChild(fallbackText);
                  }}
                />
              ) : (
                <Text className="text-gray-500 text-sm">Default LiteLLM logo will be used</Text>
              )}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3 pt-4">
            <Button onClick={handleSave} loading={loading} disabled={loading} color="indigo">
              Save Changes
            </Button>
            <Button onClick={handleReset} loading={loading} disabled={loading} variant="secondary" color="gray">
              Reset to Default
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default UIThemeSettings;
