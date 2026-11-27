import { z } from "zod";

export const skillReviewSchema = z.object({ reason: z.string() });
