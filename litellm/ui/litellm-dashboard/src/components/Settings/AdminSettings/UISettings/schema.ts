import { z } from "zod";

export const pageVisibilitySchema = z.object({ selectedPages: z.array(z.string()) });
