import { z } from "zod";

export const uiThemeSchema = z.object({
  logoUrl: z.string(),
  logoUrlDark: z.string(),
  faviconUrl: z.string(),
});
