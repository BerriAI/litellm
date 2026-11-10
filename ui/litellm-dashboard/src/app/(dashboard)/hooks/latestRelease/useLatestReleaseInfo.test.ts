import { describe, expect, it } from "vitest";
import { PENDING_RELEASE_POLL_MS, keepKnownRelease, pendingReleasePollMs } from "./useLatestReleaseInfo";

const RELEASE = { version: "1.103.0", new_features: 1, bug_fixes: 2, other_updates: 3, release_url: "https://x" };

describe("pendingReleasePollMs", () => {
  it("asks again soon while the proxy is still fetching the release", () => {
    const state = { status: "success", data: { release: null, pending: true } };
    expect(pendingReleasePollMs(state)).toBe(PENDING_RELEASE_POLL_MS);
  });

  it("keeps asking while a known release is shown and the proxy is refetching it", () => {
    const state = { status: "success", data: { release: RELEASE, pending: true } };
    expect(pendingReleasePollMs(state)).toBe(PENDING_RELEASE_POLL_MS);
  });

  it("stops once the release arrives", () => {
    expect(pendingReleasePollMs({ status: "success", data: { release: RELEASE, pending: false } })).toBe(false);
  });

  it("stops once the proxy reports the fetch failed", () => {
    expect(pendingReleasePollMs({ status: "error", data: { release: null, pending: true } })).toBe(false);
  });

  it("does not ask again when the release check is turned off", () => {
    expect(pendingReleasePollMs({ status: "success", data: { release: null, pending: false } })).toBe(false);
  });
});

describe("keepKnownRelease", () => {
  it("keeps showing the known release while the proxy refetches it", () => {
    const kept = keepKnownRelease({ release: RELEASE, pending: false }, { release: null, pending: true });
    expect(kept).toEqual({ release: RELEASE, pending: true });
  });

  it("takes a newer release over the known one", () => {
    const newer = { release: { ...RELEASE, version: "1.104.0" }, pending: false };
    expect(keepKnownRelease({ release: RELEASE, pending: true }, newer)).toBe(newer);
  });

  it("drops the known release when the release check has been turned off", () => {
    const off = { release: null, pending: false };
    expect(keepKnownRelease({ release: RELEASE, pending: false }, off)).toBe(off);
  });

  it("records a first answer as is", () => {
    const first = { release: null, pending: true };
    expect(keepKnownRelease(undefined, first)).toBe(first);
  });
});
