/**
 * Utility functions for LogDetailsDrawer component.
 * These functions handle data formatting, validation, and guardrail calculations.
 */

/**
 * Formats data for display. If input is a string, attempts to parse as JSON.
 * @param input - Data to format (string or object)
 * @returns Parsed JSON object or original input
 */
export function formatData(input: any) {
  if (typeof input === "string") {
    try {
      return JSON.parse(input);
    } catch {
      return input;
    }
  }
  return input;
}

/**
 * Checks if messages array/object contains data.
 * @param messages - Messages to check
 * @returns True if messages exist and have content
 */
export function checkHasMessages(messages: any): boolean {
  if (!messages) return false;
  if (Array.isArray(messages)) return messages.length > 0;
  if (typeof messages === "object") return Object.keys(messages).length > 0;
  return false;
}

/**
 * Checks if response object contains data.
 * @param response - Response to check
 * @returns True if response exists and has content
 */
export function checkHasResponse(response: any): boolean {
  if (!response) return false;
  return Object.keys(formatData(response)).length > 0;
}

/**
 * Normalizes guardrail information into an array.
 * @param guardrailInfo - Guardrail data (may be array, object, or null)
 * @returns Array of guardrail entries
 */
export function normalizeGuardrailEntries(guardrailInfo: any): any[] {
  if (Array.isArray(guardrailInfo)) return guardrailInfo;
  if (guardrailInfo) return [guardrailInfo];
  return [];
}

/**
 * Calculates total number of masked entities across all guardrail entries.
 * @param entries - Array of guardrail entries
 * @returns Total count of masked entities
 */
export function calculateTotalMaskedEntities(entries: any[]): number {
  return entries.reduce((sum, entry) => {
    const maskedCounts = entry?.masked_entity_count;
    if (!maskedCounts) return sum;
    return (
      sum +
      Object.values(maskedCounts).reduce<number>((acc, count) => (typeof count === "number" ? acc + count : acc), 0)
    );
  }, 0);
}

/**
 * Gets a display label for guardrail(s).
 * @param entries - Array of guardrail entries
 * @returns Display string for guardrail label
 */
export function getGuardrailLabel(entries: any[]): string {
  if (entries.length === 0) return "-";
  if (entries.length === 1) return entries[0]?.guardrail_name ?? "-";
  return `${entries.length} guardrails`;
}

/**
 * Checks if vector store data exists in metadata.
 * @param metadata - Metadata object to check
 * @returns True if vector store data exists and is non-empty
 */
export function checkHasVectorStoreData(metadata: Record<string, any>): boolean {
  return (
    metadata.vector_store_request_metadata &&
    Array.isArray(metadata.vector_store_request_metadata) &&
    metadata.vector_store_request_metadata.length > 0
  );
}
