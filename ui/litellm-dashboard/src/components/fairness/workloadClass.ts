export const DEFAULT_POOL_NAME = "default";
export const WORKLOAD_CLASS_METADATA_KEY = "priority";
export const WORKLOAD_CLASS_FIELD = "workload_class";

export const readWorkloadClass = (metadata: Record<string, unknown> | null | undefined): string => {
  const raw = metadata?.[WORKLOAD_CLASS_METADATA_KEY];
  return typeof raw === "string" && raw.length > 0 ? raw : DEFAULT_POOL_NAME;
};

export const workloadClassMetadata = (value: unknown): Record<string, string> =>
  typeof value === "string" && value.length > 0 && value !== DEFAULT_POOL_NAME
    ? { [WORKLOAD_CLASS_METADATA_KEY]: value }
    : {};

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);

export const stripWorkloadClass = <T>(metadata: T): T => {
  if (!isPlainObject(metadata)) return metadata;
  const { [WORKLOAD_CLASS_METADATA_KEY]: _dropped, ...rest } = metadata;
  return rest as T;
};

export const withWorkloadClass = <T>(metadata: T, value: unknown): T =>
  isPlainObject(metadata) && typeof value === "string"
    ? ({ ...stripWorkloadClass(metadata), ...workloadClassMetadata(value) } as T)
    : metadata;

export const withWorkloadClassJson = (metadataJson: string | undefined, value: unknown): string | undefined => {
  try {
    const parsed: unknown = JSON.parse(metadataJson?.trim() || "{}");
    if (!isPlainObject(parsed)) return metadataJson;
    const updated = withWorkloadClass(parsed, value);
    return readWorkloadClass(updated) === readWorkloadClass(parsed) ? metadataJson : JSON.stringify(updated);
  } catch {
    return metadataJson;
  }
};
