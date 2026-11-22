export type LifecycleEvent =
  | { readonly kind: "created"; readonly at: string | null }
  | { readonly kind: "lastRotated"; readonly at: string }
  | { readonly kind: "lastUpdated"; readonly at: string }
  | { readonly kind: "lastUsed"; readonly at: string | null }
  | { readonly kind: "expires"; readonly at: string | null; readonly soon: boolean; readonly expired: boolean }
  | {
      readonly kind: "nextRotation";
      readonly at: string | null;
      readonly interval: string | null;
      readonly afterExpiry: boolean;
    }
  | { readonly kind: "autoRotationOff" };

export interface LifecycleSource {
  readonly created_at?: string | null;
  readonly settings_updated_at?: string | null;
  readonly last_active?: string | null;
  readonly expires?: string | null;
  readonly auto_rotate?: boolean;
  readonly rotation_interval?: string;
  readonly last_rotation_at?: string;
  readonly key_rotation_at?: string;
  readonly next_rotation_at?: string;
}

const EXPIRY_WARNING_MS = 7 * 24 * 60 * 60 * 1000;

const timeOf = (iso: string | null): number => (iso ? Date.parse(iso) : Number.POSITIVE_INFINITY);

const validDate = (value: string | null | undefined): string | null =>
  value && !Number.isNaN(Date.parse(value)) ? value : null;

const upcomingAt = (event: LifecycleEvent): number =>
  event.kind === "expires" || event.kind === "nextRotation" ? timeOf(event.at) : Number.POSITIVE_INFINITY;

export const keyLifecycle = (key: LifecycleSource, nowMs: number): readonly LifecycleEvent[] => {
  const createdAt = validDate(key.created_at);
  const updatedAt = validDate(key.settings_updated_at);
  const expiresAt = validDate(key.expires);
  const expiryMs = timeOf(expiresAt);
  const nextRotationAt = validDate(key.next_rotation_at) ?? validDate(key.key_rotation_at);
  const lastRotationAt = validDate(key.last_rotation_at);
  const autoRotates = key.auto_rotate === true;

  const past: readonly LifecycleEvent[] = [
    { kind: "created", at: createdAt },
    ...(lastRotationAt ? [{ kind: "lastRotated" as const, at: lastRotationAt }] : []),
    ...(updatedAt && updatedAt !== createdAt ? [{ kind: "lastUpdated" as const, at: updatedAt }] : []),
    { kind: "lastUsed", at: validDate(key.last_active) },
  ];

  const expiry: LifecycleEvent = {
    kind: "expires",
    at: expiresAt,
    expired: expiryMs <= nowMs,
    soon: expiryMs > nowMs && expiryMs - nowMs <= EXPIRY_WARNING_MS,
  };
  const rotation: readonly LifecycleEvent[] =
    autoRotates || nextRotationAt
      ? [
          {
            kind: "nextRotation",
            at: nextRotationAt,
            interval: key.rotation_interval || null,
            afterExpiry: nextRotationAt !== null && timeOf(nextRotationAt) > expiryMs,
          },
        ]
      : [];
  const upcoming = [...rotation, expiry].sort((a, b) => upcomingAt(a) - upcomingAt(b));

  return [...past, ...upcoming, ...(autoRotates ? [] : [{ kind: "autoRotationOff" as const }])];
};
