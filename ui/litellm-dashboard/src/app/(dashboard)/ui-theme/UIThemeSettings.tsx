import React, { useState, useEffect, useRef } from "react";
import { Upload, File, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useTheme } from "@/contexts/ThemeContext";
import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "@/components/networking";
import { toast } from "@/lib/toast";

interface UIThemeSettingsProps {
  userID: string | null;
  userRole: string | null;
  accessToken: string | null;
}

const CSS_TEMPLATE = `/* Custom theme overrides for the LiteLLM dashboard.
   Use :root for light mode and .dark for dark mode.
   Available CSS variables:
   --background, --foreground, --card, --card-foreground,
   --popover, --popover-foreground, --primary, --primary-foreground,
   --secondary, --secondary-foreground, --muted, --muted-foreground,
   --accent, --accent-foreground, --destructive, --border, --input, --ring,
   --sidebar, --sidebar-foreground, --sidebar-border, --radius
*/
:root {
  /* Example: --primary: oklch(0.6 0.2 200); */
}

.dark {
  /* Example: --background: oklch(0.15 0.02 240); */
}
`;

const UIThemeSettings: React.FC<UIThemeSettingsProps> = ({ userID, userRole, accessToken }) => {
  const { setLogoUrl, setLogoUrlDark, setFaviconUrl, setCustomThemeCss } = useTheme();
  const [logoUrlInput, setLogoUrlInput] = useState<string>("");
  const [logoUrlDarkInput, setLogoUrlDarkInput] = useState<string>("");
  const [faviconUrlInput, setFaviconUrlInput] = useState<string>("");
  const [cssInput, setCssInput] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
        setLogoUrlInput(data.values?.logo_url || "");
        setLogoUrlDarkInput(data.values?.logo_url_dark || "");
        setFaviconUrlInput(data.values?.favicon_url || "");
        setCssInput(data.values?.custom_theme_css || "");
        setLogoUrl(data.values?.logo_url || null);
        setLogoUrlDark(data.values?.logo_url_dark || null);
        setFaviconUrl(data.values?.favicon_url || null);
        setCustomThemeCss(data.values?.custom_theme_css || null);
      }
    } catch (error) {
      console.error("Error fetching theme settings:", error);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.size > 65536) {
      toast.error("File too large. Maximum size is 64KB.");
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result;
      if (typeof content === "string") {
        setCssInput(content);
        toast.success(`Loaded "${file.name}"`);
      }
    };
    reader.readAsText(file);
    if (fileInputRef.current) fileInputRef.current.value = "";
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
          logo_url_dark: logoUrlDarkInput || null,
          favicon_url: faviconUrlInput || null,
          custom_theme_css: cssInput.trim() || null,
        }),
      });
      if (response.ok) {
        toast.success("Theme settings updated successfully!");
        setLogoUrl(logoUrlInput || null);
        setLogoUrlDark(logoUrlDarkInput || null);
        setFaviconUrl(faviconUrlInput || null);
        setCustomThemeCss(cssInput.trim() || null);
      } else {
        throw new Error("Failed to update settings");
      }
    } catch (error) {
      console.error("Error updating theme settings:", error);
      toast.fromError("Failed to update theme settings");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    setLogoUrlInput("");
    setLogoUrlDarkInput("");
    setFaviconUrlInput("");
    setCssInput("");
    setLogoUrl(null);
    setLogoUrlDark(null);
    setFaviconUrl(null);
    setCustomThemeCss(null);
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
          logo_url_dark: null,
          favicon_url: null,
          custom_theme_css: null,
        }),
      });
      if (response.ok) {
        toast.success("Theme settings reset to default!");
      } else {
        throw new Error("Failed to reset");
      }
    } catch (error) {
      console.error("Error resetting theme settings:", error);
      toast.fromError("Failed to reset theme settings");
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
        <h1 className="mb-2 text-2xl font-bold">UI Theme Customization</h1>
        <p className="text-sm text-muted-foreground">
          Customize your LiteLLM admin dashboard with a custom logo, favicon, and theme colors.
        </p>
      </div>

      <div className="space-y-6">
        <Card>
          <CardContent className="space-y-6 pt-6">
            <div>
              <Label htmlFor="ui-theme-logo-url" className="mb-2">
                Custom Logo URL
              </Label>
              <Input
                id="ui-theme-logo-url"
                placeholder="https://example.com/logo.png"
                value={logoUrlInput}
                onChange={(event) => {
                  setLogoUrlInput(event.target.value);
                  setLogoUrl(event.target.value || null);
                }}
              />
              <p className="mt-1 text-xs text-muted-foreground">
                Enter a URL for your custom logo or leave empty for default
              </p>
            </div>
            <div>
              <Label htmlFor="ui-theme-logo-url-dark" className="mb-2">
                Custom Logo URL (dark mode)
              </Label>
              <Input
                id="ui-theme-logo-url-dark"
                placeholder="https://example.com/logo-dark.png"
                value={logoUrlDarkInput}
                onChange={(event) => {
                  setLogoUrlDarkInput(event.target.value);
                  setLogoUrlDark(event.target.value || null);
                }}
              />
              <p className="mt-1 text-xs text-muted-foreground">
                Enter a URL for a logo suited to dark backgrounds, or leave empty to reuse the logo above
              </p>
            </div>
            <div>
              <Label htmlFor="ui-theme-favicon-url" className="mb-2">
                Custom Favicon URL
              </Label>
              <Input
                id="ui-theme-favicon-url"
                placeholder="https://example.com/favicon.ico"
                value={faviconUrlInput}
                onChange={(event) => {
                  setFaviconUrlInput(event.target.value);
                  setFaviconUrl(event.target.value || null);
                }}
              />
              <p className="mt-1 text-xs text-muted-foreground">
                Enter a URL for your custom favicon (.ico, .png, or .svg) or leave empty for default
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-4 pt-6">
            <div>
              <div className="flex items-center justify-between mb-2">
                <Label htmlFor="ui-theme-css">
                  Custom Theme CSS
                </Label>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <Upload className="size-3.5 mr-1.5" />
                    Upload CSS File
                  </Button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".css,.txt"
                    className="hidden"
                    onChange={handleFileUpload}
                  />
                  {cssInput && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setCssInput("")}
                      title="Clear custom CSS"
                    >
                      <X className="size-3.5 mr-1.5" />
                      Clear
                    </Button>
                  )}
                </div>
              </div>
              <Textarea
                id="ui-theme-css"
                placeholder={CSS_TEMPLATE}
                value={cssInput}
                onChange={(event) => setCssInput(event.target.value)}
                rows={16}
                className="font-mono text-xs resize-y min-h-[200px]"
              />
              <p className="mt-1 text-xs text-muted-foreground">
                Override theme CSS variables. Use <code className="px-1 rounded bg-muted">:root</code> for
                light mode and <code className="px-1 rounded bg-muted">.dark</code> for dark mode. Changes
                apply live as you type.
              </p>
            </div>
          </CardContent>
        </Card>

        <div className="flex gap-3">
          <Button onClick={handleSave} disabled={loading}>
            {loading && <UiLoadingSpinner className="size-4" />}
            Save Changes
          </Button>
          <Button variant="outline" onClick={handleReset} disabled={loading}>
            {loading && <UiLoadingSpinner className="size-4" />}
            Reset to Default
          </Button>
        </div>
      </div>
    </div>
  );
};

export default UIThemeSettings;
