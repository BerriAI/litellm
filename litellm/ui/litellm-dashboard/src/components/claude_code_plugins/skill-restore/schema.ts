import { z } from "zod";

export const skillRestoreSchema = z.object({ newVersion: z.string() });
