/**
 * The admin-owned logging destinations assigned to an identity (key / team / org).
 *
 * Assignments live on a typed logging_exporters column, surfaced at the top level of
 * the API object. A metadata.logging_exporters fallback is kept only so a row written
 * before the column existed still renders; the column is the source of truth.
 */
export const loggingExportersOf = (obj: unknown): string[] => {
  const record = obj as {
    logging_exporters?: unknown;
    metadata?: { logging_exporters?: unknown } | null;
  } | null;
  if (Array.isArray(record?.logging_exporters)) {
    return record.logging_exporters as string[];
  }
  const fromMetadata = record?.metadata?.logging_exporters;
  return Array.isArray(fromMetadata) ? (fromMetadata as string[]) : [];
};
