export type LensPolicy = boolean | null;

export function readLensPolicy(metadata: Readonly<Record<string, unknown>> | null | undefined): LensPolicy {
  const policy = metadata?.lens_enabled;
  return typeof policy === "boolean" ? policy : null;
}

export function effectiveLensEnabled(policy: LensPolicy, teamPolicy: LensPolicy, configured: boolean): boolean {
  if (policy !== null) return policy;
  if (teamPolicy !== null) return teamPolicy;
  return configured;
}

export function lensCreationMetadata(
  metadata: Readonly<Record<string, unknown>>,
  policy: LensPolicy | undefined,
): Record<string, unknown> {
  if (policy === null || policy === undefined) return { ...metadata };
  return { ...metadata, lens_enabled: policy };
}

export function readLensPolicyJson(metadata: unknown): LensPolicy {
  if (typeof metadata !== "string" || !metadata) return null;
  try {
    const parsed: unknown = JSON.parse(metadata);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return null;
    return readLensPolicy(parsed as Readonly<Record<string, unknown>>);
  } catch {
    return null;
  }
}
