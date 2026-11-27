import { z } from "zod";

import i18n from "@/i18n/i18n";

export const SECRET_NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/;

export const secretFormSchema = z.object({
  name: z.string().regex(SECRET_NAME_PATTERN, { error: () => i18n.t("secretsVault.modal.errors.name") }),
  value: z.string().min(1, { error: () => i18n.t("secretsVault.modal.errors.value") }),
  description: z.string().max(500),
});

export type SecretFormValues = z.output<typeof secretFormSchema>;

export const EMPTY_SECRET: SecretFormValues = { name: "", value: "", description: "" };

export const toSecretWrite = (values: SecretFormValues) => ({
  value: values.value,
  description: values.description.trim() || null,
});
