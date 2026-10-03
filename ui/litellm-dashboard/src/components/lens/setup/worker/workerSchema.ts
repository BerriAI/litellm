import { z } from "zod";

export const workerAddressSchema = z.string().superRefine((address, ctx) => {
  try {
    const parsed = new URL(address);
    if (!["http:", "https:"].includes(parsed.protocol) || parsed.username || parsed.password) {
      ctx.addIssue({ code: "custom", message: "Enter an HTTP or HTTPS proxy URL without credentials" });
    }
  } catch (error) {
    ctx.addIssue({ code: "custom", message: error instanceof Error ? error.message : "Invalid URL" });
  }
});

export function validateWorkerAddress(address: string) {
  const result = workerAddressSchema.safeParse(address);
  if (!result.success) throw new Error(result.error.issues[0].message);
}

const analysisAccessFields = { model: z.string().nullable(), budget: z.string() };
export const analysisAccessSchema = z
  .object(analysisAccessFields)
  .superRefine((access, ctx) => {
    if (!access.model || !Number.isFinite(Number(access.budget)) || Number(access.budget) <= 0)
      ctx.addIssue({ code: "custom", message: "Choose a model and a monthly limit greater than zero" });
  })
  .transform((access) => ({ model: access.model ?? "", budget: Number(access.budget) }));

export type AnalysisAccess = z.input<typeof analysisAccessSchema>;
