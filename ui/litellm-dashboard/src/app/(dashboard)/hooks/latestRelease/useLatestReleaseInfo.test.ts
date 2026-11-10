import { describe, expect, it } from "vitest";
import { PENDING_RELEASE_MAX_ANSWERS, PENDING_RELEASE_POLL_MS, pendingReleasePollMs } from "./useLatestReleaseInfo";

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
    expect(pendingReleasePollMs({ status: "success", data: null, dataUpdateCount: PENDING_RELEASE_MAX_ANSWERS })).toBe(
      false,
    );
  });
});
