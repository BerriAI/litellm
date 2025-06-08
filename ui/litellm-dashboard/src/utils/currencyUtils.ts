// currencyUtils.ts
//
// Utility functions for formatting and working with currency in the application.
// These functions rely on locale and currency settings typically provided by the backend
// and made available via imports from a configuration module (e.g., networking.ts).
//
// Usage:
// - formatCurrency(amount): Formats a number as a currency string.
// - getCurrencySymbol(): Retrieves the symbol of the current currency.
// - getCurrencyCode(): Returns the current currency code from the config.
//
// Dependencies:
// - localeIsoCode and currencyCode are expected to be dynamically set at runtime,
//   typically by loading them once from the backend and storing them in a shared module.

import { localeIsoCode, currencyCode } from "../components/networking";

/**
 * Formats a number or string as a localized currency string.
 *
 * - Automatically uses configured locale and currency if none provided.
 * - Safely handles `null`, `undefined`, and invalid numeric inputs.
 *
 * @param amount - The numeric value to format (can be number or string).
 * @param locale - Optional. Overrides the default locale for formatting.
 * @param currency - Optional. Overrides the default currency for formatting.
 * @returns The formatted currency string, or an empty string if input is invalid.
 *
 * @example
 * formatCurrency(1000) → "CHF 1’000.00" (if locale is "de-CH" and currency is "CHF")
 */
export const formatCurrency = (
  amount: number | string | null,
  locale: string = localeIsoCode,
  currency: string = currencyCode
): string => {
  if (amount === undefined || amount === null) return "";
  const numeric = typeof amount === "string" ? parseFloat(amount) : amount;
  if (isNaN(numeric)) return "";

  return new Intl.NumberFormat(locale, {
    style: "currency",
    currency,
  }).format(numeric);
};

/**
 * Retrieves the currency symbol for the current or provided locale and currency.
 *
 * - Uses a clever trick with `0` to get only the currency symbol.
 * - Returns a trimmed string without any digits.
 *
 * @param locale - Optional. Overrides the default locale.
 * @param currency - Optional. Overrides the default currency.
 * @returns The currency symbol (e.g., "CHF", "$", "€").
 *
 * @example
 * getCurrencySymbol() → "CHF" (if locale is "de-CH" and currency is "CHF")
 */
export const getCurrencySymbol = (
  locale: string = localeIsoCode,
  currency: string = currencyCode
): string => {
  return (0)
    .toLocaleString(locale, {
      style: "currency",
      currency,
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    })
    .replace(/\d/g, "")
    .trim();
};

/**
 * Returns the current currency code from the application settings.
 *
 * @returns The ISO currency code string (e.g., "CHF", "USD", "EUR").
 *
 * @example
 * getCurrencyCode() → "CHF"
 */
export const getCurrencyCode = (): string => {
  return currencyCode;
};
