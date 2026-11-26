import { z } from "zod";

const disableableCapabilitySchema = z.enum(["prompts", "resources"]);
export type DisableableCapability = z.infer<typeof disableableCapabilitySchema>;

const clientDraftSchema = z.object({
  key: z.string().nullable(),
  alias: z.string(),
  value: z.string(),
});
export type ClientDraft = z.infer<typeof clientDraftSchema>;

const networkSettingsFields = {
  privateRanges: z.custom<string[]>(Array.isArray),
  allowedClients: z.array(clientDraftSchema.extend({ key: z.string() })),
  clientIdHeader: z.string(),
  disabledCapabilities: z.array(disableableCapabilitySchema),
  rangeDraft: z.string(),
  clientDraft: clientDraftSchema.nullable(),
};

export const mcpNetworkSettingsSchema = z.object(networkSettingsFields);

export const defaultNetworkSettings: z.input<typeof mcpNetworkSettingsSchema> = {
  privateRanges: [],
  allowedClients: [],
  clientIdHeader: "",
  disabledCapabilities: [],
  rangeDraft: "",
  clientDraft: null,
};
