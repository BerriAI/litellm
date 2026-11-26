import { z } from "zod";

export const editCustomerSchema = z.object({
  alias: z.string(),
  budget: z.string(),
});

export type EditCustomerFormValues = z.output<typeof editCustomerSchema>;
