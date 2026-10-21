import type { KeyMetadata } from "./types";

export function keyActivityLabel(
  metadata: Pick<KeyMetadata, "key_alias" | "user_alias" | "user_email"> | null | undefined,
  fallback = "-",
): string {
  const byAlias = metadata?.key_alias || metadata?.user_alias;
  return byAlias || metadata?.user_email || fallback;
}
