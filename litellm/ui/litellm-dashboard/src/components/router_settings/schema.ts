import { z } from "zod";
import type { RouterSettingsFormValue } from "./RouterSettingsForm";

export const routerSettingsSchema = z.object({
  selectedStrategy: z.custom<RouterSettingsFormValue["selectedStrategy"]>(),
  enableTagFiltering: z.custom<RouterSettingsFormValue["enableTagFiltering"]>(),
});
