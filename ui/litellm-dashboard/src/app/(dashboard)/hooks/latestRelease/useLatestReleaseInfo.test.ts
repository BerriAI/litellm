import { describe, expect, it } from "vitest";
import {
  PENDING_RELEASE_ANSWERS_PER_BURST,
  PENDING_RELEASE_POLL_MS,
  keepKnownRelease,
  pendingReleasePollMs,
} from "./useLatestReleaseInfo";

const RELEASE = { version: "1.103.0", new_features: 1, bug_fixes: 2, other_updates: 3, release_url: "https://x" };

describe("pendingReleasePollMs", () => {
  it("asks again soon while the proxy is still fetching the release", () => {
    expect(pendingReleasePollMs({ status: "success", data: null, dataUpdateCount: 1 })).toBe(PENDING_RELEASE_POLL_MS);
  });

  it("stops once the release arrives", () => {
    expect(pendingReleasePollMs({ status: "success", data: RELEASE, dataUpdateCount: 2 })).toBe(false);
  });

  it("stops once the proxy reports the fetch failed", () => {
    expect(pendingReleasePollMs({ status: "error", data: null, dataUpdateCount: 1 })).toBe(false);
  });

  it("stops after a few empty answers, as when the release check is turned off", () => {
    const state = { status: "success", data: null, dataUpdateCount: PENDING_RELEASE_ANSWERS_PER_BURST };
    expect(pendingReleasePollMs(state)).toBe(false);
  });

  it("asks again soon when a later refetch finds the proxy fetching once more", () => {
    const state = { status: "success", data: null, dataUpdateCount: PENDING_RELEASE_ANSWERS_PER_BURST + 1 };
    expect(pendingReleasePollMs(state)).toBe(PENDING_RELEASE_POLL_MS);
  });
});

describe("keepKnownRelease", () => {
  it("keeps showing the known release while the proxy refetches it", () => {
    expect(keepKnownRelease(RELEASE, null)).toBe(RELEASE);
  });

  it("takes a newer release over the known one", () => {
    const newer = { ...RELEASE, version: "1.104.0" };
    expect(keepKnownRelease(RELEASE, newer)).toBe(newer);
  });

  it("records an empty first answer so the dashboard asks again", () => {
    expect(keepKnownRelease(undefined, null)).toBeNull();
  });
});
