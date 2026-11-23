import type { ParseKeys } from "i18next";
import i18n from "@/i18n/i18n";
import { isMaskedSecret } from "@/utils/maskedSecretUtils";

export interface FederationField {
  readonly key: string;
  readonly labelKey: ParseKeys;
  readonly tooltipKey: ParseKeys;
  readonly tooltipValues?: Readonly<Record<string, number>>;
  readonly placeholder?: string;
  readonly required: boolean;
  readonly control: "text" | "integer" | "select";
  readonly options?: readonly string[];
}

export const isBlank = (value: unknown): boolean => {
  if (typeof value === "string") {
    return value.trim() === "";
  }
  return value === undefined || value === null;
};

export const requiredFederationValue = (value: unknown): string | true =>
  isBlank(value) ? i18n.t("modelAdd.federationField.required") : true;

export const validateMaskedValueUntouched =
  (storedValue: unknown) =>
  (value: unknown): string | true =>
    isMaskedSecret(value) && value !== storedValue ? i18n.t("modelAdd.federationField.maskedValueEdited") : true;
