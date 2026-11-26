import { z } from "zod";

export const connectorImportSchema = z.object({
  configText: z.string(),
});
