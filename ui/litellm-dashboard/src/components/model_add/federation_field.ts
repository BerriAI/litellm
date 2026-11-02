import { isMaskedSecret } from "@/utils/maskedSecretUtils";

export interface FederationField {
  readonly key: string;
  readonly label: string;
  readonly tooltip: string;
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

export const requiredFederationValue = (value: unknown): string | true => (isBlank(value) ? "Required" : true);

export const validateMaskedValueUntouched =
  (storedValue: unknown) =>
  (value: unknown): string | true =>
    isMaskedSecret(value) && value !== storedValue
      ? "This stored value is hidden. Replace the whole value to change it"
      : true;
