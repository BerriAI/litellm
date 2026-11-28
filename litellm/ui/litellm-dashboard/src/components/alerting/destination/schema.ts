import { z } from "zod";

export const destinationVariablesSchema = z.record(z.string(), z.string());

export type DestinationVariablesFields = z.infer<typeof destinationVariablesSchema>;
