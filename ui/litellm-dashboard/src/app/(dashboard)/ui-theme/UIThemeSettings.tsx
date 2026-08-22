import React, { useState, useEffect, useRef, useCallback } from "react";
import { Upload, X, ChevronDown, ChevronUp, Palette } from "lucide-react";
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

interface ColorVar {
  key: string;
  label: string;
  description: string;
}

const LIGHT_VARS: ColorVar[] = [
  { key: "background", label: "Background", description: "Page background" },
  { key: "foreground", label: "Text", description: "Primary text color" },
  { key: "card", label: "Card", description: "Card & panel background" },
  { key: "card-foreground", label: "Card Text", description: "Text on cards" },
  { key: "primary", label: "Primary", description: "Buttons & actions" },
  { key: "primary-foreground", label: "Primary Text", description: "Text on primary buttons" },
  { key: "muted", label: "Muted", description: "Subtle backgrounds" },
  { key: "muted-foreground", label: "Muted Text", description: "Secondary text" },
  { key: "accent", label: "Accent", description: "Highlights & hover" },
  { key: "border", label: "Border", description: "Borders & dividers" },
];

const DARK_VARS: ColorVar[] = [
  { key: "background", label: "Background", description: "Page background" },
  { key: "foreground", label: "Text", description: "Primary text color" },
  { key: "card", label: "Card", description: "Card & panel background" },
  { key: "card-foreground", label: "Card Text", description: "Text on cards" },
  { key: "primary", label: "Primary", description: "Buttons & actions" },
  { key: "primary-foreground", label: "Primary Text", description: "Text on primary buttons" },
  { key: "muted", label: "Muted", description: "Subtle backgrounds" },
  { key: "muted-foreground", label: "Muted Text", description: "Secondary text" },
  { key: "accent", label: "Accent", description: "Highlights & hover" },
  { key: "border", label: "Border", description: "Borders & dividers" },
];

function generateCss(lightColors: Record<string, string>, darkColors: Record<string, string>, advancedCss: string): string {
  const parts: string[] = [];
  const lightRules = Object.entries(lightColors).filter(([, v]) => v);
  const darkRules = Object.entries(darkColors).filter(([, v]) => v);

  if (lightRules.length > 0) {
    parts.push(`:root {\n  ${lightRules.map(([k, v]) => `--${k}: ${v};`).join("\n  ")}\n}`);
  }
  if (darkRules.length > 0) {
    parts.push(`.dark {\n  ${darkRules.map(([k, v]) => `--${k}: ${v};`).join("\n  ")}\n}`);
  }
  if (advancedCss.trim()) {
    parts.push(advancedCss.trim());
  }
  return parts.join("\n\n");
}

const UIThemeSettings: React.FC<UIThemeSettingsProps> = ({ userID, userRole, accessToken }) => {
  const { setLogoUrl, setLogoUrlDark, setFaviconUrl, setCustomThemeCss } = useTheme();
  const [logoUrlInput, setLogoUrlInput] = useState<string>("");
  const [logoUrlDarkInput, setLogoUrlDarkInput] = useState<string>("");
  const [faviconUrlInput, setFaviconUrlInput] = useState<string>("");
  const [lightColors, setLightColors] = useState<Record<string, string>>({});
  const [darkColors, setDarkColors] = useState<Record<string, string>>({});
  const [advancedCss, setAdvancedCss] = useState<string>("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (accessToken) {
      fetchThemeSettings();
    }
  }, [accessToken]);

  const parseCssToColors = (css: string): { light: Record<string, string>, dark: Record<string, string>, extra: string } => {
    const light: Record<string, string> = {};
    const dark: Record<string, string> = {};
    const extraParts: string[] = [];

    const rootMatch = css.match(/:root\s*\{([^}]+)\}/g);
    const darkMatch = css.match(/\.dark\s*\{([^}]+)\}/g);

    if (rootMatch) {
      for (const block of rootMatch) {
        const inner = block.replace(/:root\s*\{/, "").replace(/\}/, "");
        for (const line of inner.split(";")) {
          const match = line.match(/--([\w-]+)\s*:\s*([^;]+)/);
          if (match) {
            if (match[1].startsWith("color") || LIGHT_VARS.some(v => v.key === match[1])) {
              light[match[1]] = match[2].trim();
            } else {
              extraParts.push(`:root { --${match[1]}: ${match[2].trim()}; }`);
            }
          }
        }
      }
    }

    if (darkMatch) {
      for (const block of darkMatch) {
        const inner = block.replace(/\.dark\s*\{/, "").replace(/\}/, "");
        for (const line of inner.split(";")) {
          const match = line.match(/--([\w-]+)\s*:\s*([^;]+)/);
          if (match) {
            if (match[1].startsWith("color") || DARK_VARS.some(v => v.key === match[1])) {
              dark[match[1]] = match[2].trim();
            } else {
              extraParts.push(`.dark { --${match[1]}: ${match[2].trim()}; }`);
            }
          }
        }
      }
    }

    return { light, dark, extra: extraParts.join("\n") };
  };

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
        setLogoUrl(data.values?.logo_url || null);
        setLogoUrlDark(data.values?.logo_url_dark || null);
        setFaviconUrl(data.values?.favicon_url || null);

        const rawCss = data.values?.custom_theme_css || "";
        if (rawCss) {
          const parsed = parseCssToColors(rawCss);
          setLightColors(parsed.light);
          setDarkColors(parsed.dark);
          setAdvancedCss(parsed.extra);
          setCustomThemeCss(rawCss);
        }
      }
    } catch (error) {
      console.error("Error fetching theme settings:", error);
    }
  };

  const updateColor = useCallback((mode: "light" | "dark", key: string, value: string) => {
    const setColors = mode === "light" ? setLightColors : setDarkColors;
    setColors((prev) => {
      const next = { ...prev, [key]: value };
      const css = generateCss(
        mode === "light" ? next : lightColors,
        mode === "dark" ? next : darkColors,
        advancedCss
      );
      setCustomThemeCss(css || null);
      return next;
    });
  }, [lightColors, darkColors, advancedCss, setCustomThemeCss]);

  const handleAdvancedChange = (value: string) => {
    setAdvancedCss(value);
    const css = generateCss(lightColors, darkColors, value);
    setCustomThemeCss(css || null);
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
        const parsed = parseCssToColors(content);
        setLightColors(parsed.light);
        setDarkColors(parsed.dark);
        setAdvancedCss(parsed.extra);
        const css = generateCss(parsed.light, parsed.dark, parsed.extra);
        setCustomThemeCss(css || null);
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
      const css = generateCss(lightColors, darkColors, advancedCss);
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
          custom_theme_css: css || null,
        }),
      });
      if (response.ok) {
        toast.success("Theme settings updated successfully!");
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
    setLightColors({});
    setDarkColors({});
    setAdvancedCss("");
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

  const ColorGrid: React.FC<{
    title: string;
    vars: ColorVar[];
    colors: Record<string, string>;
    onChange: (key: string, value: string) => void;
  }> = ({ title, vars, colors, onChange }) => (
    <div>
      <h3 className="mb-3 text-sm font-semibold">{title}</h3>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-5">
        {vars.map((v) => (
          <div key={v.key} className="group flex flex-col items-center gap-1.5 rounded-lg border p-2">
            <div className="relative h-8 w-full overflow-hidden rounded-md border" style={{ backgroundColor: colors[v.key] || "transparent" }}>
              <input
                type="color"
                className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
                value={colors[v.key] || "#ffffff"}
                onChange={(e) => onChange(v.key, e.target.value)}
              />
              {colors[v.key] && (
                <button
                  className="absolute right-0.5 top-0.5 rounded-full bg-black/30 p-0.5 text-white opacity-0 transition-opacity group-hover:opacity-100"
                  onClick={() => onChange(v.key, "")}
                  title="Reset"
                >
                  <X className="size-3" />
                </button>
              )}
            </div>
            <span className="text-center text-[10px] font-medium leading-tight">{v.label}</span>
          </div>
        ))}
      </div>
    </div>
  );

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full mx-auto max-w-4xl px-6 py-8">
      <div className="mb-8">
        <h1 className="mb-2 text-2xl font-bold">Theme Customization</h1>
        <p className="text-sm text-muted-foreground">
          Customize your LiteLLM dashboard colors, logo, and favicon.
        </p>
      </div>

      <div className="space-y-6">
        <Card>
          <CardContent className="space-y-6 pt-6">
            <div className="flex items-center gap-2">
              <Palette className="size-4" />
              <h2 className="text-sm font-semibold">Theme Colors</h2>
            </div>
            <p className="text-xs text-muted-foreground">
              Pick colors to override the default theme. Click a swatch to choose a color, or click the X on a swatch to reset it. Changes apply live.
            </p>

            <ColorGrid
              title="Light Mode"
              vars={LIGHT_VARS}
              colors={lightColors}
              onChange={(key, value) => updateColor("light", key, value)}
            />

            <ColorGrid
              title="Dark Mode"
              vars={DARK_VARS}
              colors={darkColors}
              onChange={(key, value) => updateColor("dark", key, value)}
            />
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-4 pt-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Upload className="size-4" />
                <h2 className="text-sm font-semibold">Logo & Favicon</h2>
              </div>
              <Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()}>
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
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <Label htmlFor="ui-theme-logo-url" className="mb-2 text-xs">Custom Logo (light)</Label>
                <Input
                  id="ui-theme-logo-url"
                  placeholder="https://example.com/logo.png"
                  value={logoUrlInput}
                  onChange={(event) => {
                    setLogoUrlInput(event.target.value);
                    setLogoUrl(event.target.value || null);
                  }}
                />
              </div>
              <div>
                <Label htmlFor="ui-theme-logo-url-dark" className="mb-2 text-xs">Custom Logo (dark)</Label>
                <Input
                  id="ui-theme-logo-url-dark"
                  placeholder="https://example.com/logo-dark.png"
                  value={logoUrlDarkInput}
                  onChange={(event) => {
                    setLogoUrlDarkInput(event.target.value);
                    setLogoUrlDark(event.target.value || null);
                  }}
                />
              </div>
              <div className="sm:col-span-2">
                <Label htmlFor="ui-theme-favicon-url" className="mb-2 text-xs">Custom Favicon</Label>
                <Input
                  id="ui-theme-favicon-url"
                  placeholder="https://example.com/favicon.ico"
                  value={faviconUrlInput}
                  onChange={(event) => {
                    setFaviconUrlInput(event.target.value);
                    setFaviconUrl(event.target.value || null);
                  }}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <button
              className="flex w-full items-center justify-between"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <span className="text-sm font-semibold">Advanced CSS</span>
              {showAdvanced ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
            </button>
            {showAdvanced && (
              <div className="mt-3">
                <Textarea
                  placeholder={`/* Additional CSS rules beyond the color pickers above */\n/* Example: */\n/* :root { --radius: 1rem; } */`}
                  value={advancedCss}
                  onChange={(event) => handleAdvancedChange(event.target.value)}
                  rows={8}
                  className="font-mono text-xs resize-y min-h-[120px]"
                />
                <p className="mt-1 text-xs text-muted-foreground">
                  For anything the color pickers don't cover: radius, shadows, layout tweaks, or non-color variables.
                </p>
              </div>
            )}
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
