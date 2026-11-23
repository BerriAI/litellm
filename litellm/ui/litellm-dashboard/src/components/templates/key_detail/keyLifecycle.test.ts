import { describe, expect, it } from "vitest";

import { keyLifecycle } from "./keyLifecycle";

const NOW = Date.parse("2026-11-21T12:00:00Z");

const NEVER_EXPIRES = { kind: "expires", at: null, soon: false, expired: false } as const;

const kinds = (events: ReturnType<typeof keyLifecycle>) => events.map((event) => event.kind);

describe("keyLifecycle", () => {
  it("lists a never-used, never-rotated key as created, unused, never expiring and not rotating", () => {
    const events = keyLifecycle({ created_at: "2026-06-11T09:00:00Z" }, NOW);

    expect(kinds(events)).toEqual(["created", "lastUsed", "expires", "autoRotationOff"]);
    expect(events).toContainEqual({ kind: "lastUsed", at: null });
    expect(events).toContainEqual(NEVER_EXPIRES);
  });

  it("shows the last configuration change only when it differs from creation", () => {
    const created = "2026-06-11T09:00:00Z";

    expect(kinds(keyLifecycle({ created_at: created, settings_updated_at: created }, NOW))).not.toContain(
      "lastUpdated",
    );
    expect(keyLifecycle({ created_at: created, settings_updated_at: "2026-11-19T16:12:00Z" }, NOW)).toContainEqual({
      kind: "lastUpdated",
      at: "2026-11-19T16:12:00Z",
    });
  });

  it("orders past events before upcoming ones, with upcoming events sorted by date", () => {
    const rotatingKey = {
      created_at: "2026-10-02T10:05:00Z",
      settings_updated_at: "2026-11-19T16:12:00Z",
      last_rotation_at: "2026-11-01T00:00:00Z",
      last_active: "2026-11-21T11:58:00Z",
      expires: "2027-03-01T00:00:00Z",
      auto_rotate: true,
      rotation_interval: "30d",
      next_rotation_at: "2026-12-01T00:00:00Z",
    };
    const events = keyLifecycle(rotatingKey, NOW);

    expect(kinds(events)).toEqual(["created", "lastRotated", "lastUpdated", "lastUsed", "nextRotation", "expires"]);
  });

  it("puts an expiry that comes first ahead of the next rotation and marks the rotation as after expiry", () => {
    const expiresFirst = {
      created_at: "2026-10-02T10:05:00Z",
      expires: "2026-11-27T00:00:00Z",
      auto_rotate: true,
      rotation_interval: "30d",
      next_rotation_at: "2026-12-01T00:00:00Z",
    };
    const events = keyLifecycle(expiresFirst, NOW);
    const rotation = { kind: "nextRotation", at: "2026-12-01T00:00:00Z", interval: "30d", afterExpiry: true };

    expect(kinds(events).slice(-2)).toEqual(["expires", "nextRotation"]);
    expect(events).toContainEqual(rotation);
  });

  it("warns about an expiry within a week and flags one that has passed", () => {
    const soon = keyLifecycle({ expires: "2026-11-27T00:00:00Z" }, NOW).find((event) => event.kind === "expires");
    const later = keyLifecycle({ expires: "2026-12-05T00:00:00Z" }, NOW).find((event) => event.kind === "expires");
    const past = keyLifecycle({ expires: "2026-11-20T00:00:00Z" }, NOW).find((event) => event.kind === "expires");

    expect(soon).toMatchObject({ soon: true, expired: false });
    expect(later).toMatchObject({ soon: false, expired: false });
    expect(past).toMatchObject({ soon: false, expired: true });
  });

  it("keeps a scheduled rotation from the legacy key_rotation_at field", () => {
    const events = keyLifecycle({ key_rotation_at: "2026-12-10T00:00:00Z" }, NOW);

    const rotation = { kind: "nextRotation", at: "2026-12-10T00:00:00Z", interval: null, afterExpiry: false };
    expect(events).toContainEqual(rotation);
    expect(kinds(events)).toContain("autoRotationOff");
  });

  it("shows a rotating key with no date yet as a pending rotation, not as rotation off", () => {
    const events = keyLifecycle({ auto_rotate: true, rotation_interval: "90d" }, NOW);

    const pending = { kind: "nextRotation", at: null, interval: "90d", afterExpiry: false };
    expect(events).toContainEqual(pending);
    expect(kinds(events)).not.toContain("autoRotationOff");
  });

  it("treats unparseable dates as missing instead of passing them on", () => {
    const events = keyLifecycle({ created_at: "2026-06-11T09:00:00Z", expires: "null", last_active: "never" }, NOW);

    expect(events).toContainEqual(NEVER_EXPIRES);
    expect(events).toContainEqual({ kind: "lastUsed", at: null });
  });
});
