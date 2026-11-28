import { z } from "zod";

export const internalAccessSchema = z.object({ draft: z.array(z.string()).nullable() });
