/**
 * Utility functions for handling key information and metadata
 */

// Fields that should be filtered out from metadata display due to security concerns
const SENSITIVE_METADATA_FIELDS = ["logging"] as const;

/**
 * Filters out sensitive information from metadata for display purposes
 * @param metadata - The metadata object to filter
 * @returns Filtered metadata object without sensitive fields
 */
export const filterSensitiveMetadata = (metadata: Record<string, any> | null | undefined): Record<string, any> => {
  if (!metadata || typeof metadata !== "object") {
    return {};
  }

  return Object.fromEntries(
    Object.entries(metadata).filter(([key]) => !SENSITIVE_METADATA_FIELDS.includes(key as any)),
  );
};

/**
 * Safely extracts logging settings from metadata
 * @param metadata - The metadata object to extract from
 * @returns Array of logging configurations or empty array if not found
 */
export const extractLoggingSettings = (metadata: Record<string, any> | null | undefined): any[] => {
  if (!metadata || typeof metadata !== "object") {
    return [];
  }

  return Array.isArray(metadata.logging) ? metadata.logging : [];
};

/**
 * Formats metadata for JSON display (with sensitive fields filtered)
 * @param metadata - The metadata object to format
 * @param indent - Number of spaces for indentation (default: 2)
 * @returns Formatted JSON string
 */
export const formatMetadataForDisplay = (
  metadata: Record<string, any> | null | undefined,
  indent: number = 2,
): string => {
  const filtered = filterSensitiveMetadata(metadata);
  return JSON.stringify(filtered, null, indent);
};

/**
 * Removes the top-level "tags" property from a metadata object.
 * This prevents duplicated tag information in UIs where tags are managed separately.
 * @param metadata - The metadata value to process; returned as-is if not an object
 * @returns A shallow copy of the object without the "tags" key, or the original value for non-objects
 */
export const stripTagsFromMetadata = (metadata: any) => {
  if (!metadata || typeof metadata !== "object") {
    return metadata;
  }
  // Remove tags key from metadata shown in textarea to avoid duplication
  const { tags, ...rest } = metadata as Record<string, any>;
  return rest;
};
