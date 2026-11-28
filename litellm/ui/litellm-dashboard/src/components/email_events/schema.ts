import { z } from "zod";
import type { EmailEventSetting } from "./types";

export const emailEventSettingsSchema = z.object({
  settings: z.array(z.custom<EmailEventSetting>()),
});

export type EmailEventSettingsFields = z.infer<typeof emailEventSettingsSchema>;
