import type { z } from "zod";
import type { uiThemeSchema } from "./schema";

type UIThemeFields = z.infer<typeof uiThemeSchema>;

interface SavedThemeImages {
  logo_url?: string | null;
  logo_url_dark?: string | null;
  favicon_url?: string | null;
}

export const toUIThemeFields = (images?: SavedThemeImages | null): UIThemeFields => ({
  logoUrl: images?.logo_url || "",
  logoUrlDark: images?.logo_url_dark || "",
  faviconUrl: images?.favicon_url || "",
});

export const toUIThemeUpdate = ({ logoUrl, logoUrlDark, faviconUrl }: UIThemeFields) => ({
  logo_url: logoUrl || null,
  logo_url_dark: logoUrlDark || null,
  favicon_url: faviconUrl || null,
});
