import { z } from "zod";
import { switchyardSaveError, type SwitchyardConfig } from "../switchyard_config";

export const switchyardEditSchema = z
  .object({ name: z.string(), config: z.custom<SwitchyardConfig>() })
  .superRefine(({ name, config }, ctx) => {
    const error = switchyardSaveError(name, config);
    if (error) ctx.addIssue({ code: "custom", message: error });
  });

export type SwitchyardEditValues = z.infer<typeof switchyardEditSchema>;
