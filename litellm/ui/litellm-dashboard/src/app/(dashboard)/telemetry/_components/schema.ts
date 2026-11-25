import { z } from "zod";
import { PROXY_GROUPS, UI_GROUPS } from "./telemetryGroups";

export const telemetryUpdateSchema = z.object({
  groups: z.array(z.enum([...PROXY_GROUPS, ...UI_GROUPS])),
  destination: z.enum(["local_table", "http"]).nullable(),
});
