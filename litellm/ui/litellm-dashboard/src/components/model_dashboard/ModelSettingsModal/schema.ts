import { z } from "zod";

export const modelSettingsSchema = z.object({ store_model_in_db: z.unknown() });

export type ModelSettingsFields = z.infer<typeof modelSettingsSchema>;
