import { z } from "zod";

export const isBannerMessageMissing = ({ enabled, message }: { enabled: boolean; message: string }) =>
  enabled && message.trim() === "";

export const userBannerSchema = z
  .object({
    enabled: z.boolean(),
    message: z.string(),
    severity: z.enum(["info", "warning", "error"]),
  })
  .refine((draft) => !isBannerMessageMissing(draft), { path: ["message"] });
