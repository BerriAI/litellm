import { renderWithProviders, screen, within } from "../../../../tests/test-utils";
import { AUTO_ROUTER_DOCS_URL, NotificationsBell, NotificationsBellView } from "./NotificationsBell";
import type { LatestReleaseInfo } from "@/app/(dashboard)/hooks/latestRelease/useLatestReleaseInfo";
import React from "react";
import type { SavedAlert } from "@/components/Alerts/alertPresentation";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/components/ModelDeprecationAlerts/useDeprecationAlerts", () => ({
  useDeprecationAlerts: () => ({
    data: [],
    scope: "test",
    range: { start_date: "2026-10-06", end_date: "2026-11-04" },
  }),
}));

vi.mock("@/app/(dashboard)/hooks/currentUser/useCurrentUser", () => ({
  useCurrentUser: () => ({ data: undefined }),
}));
vi.mock("@/components/Alerts/useAlerts", () => ({
  useAlerts: () => ({ data: undefined }),
}));

vi.mock("@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails", () => ({
  useHealthReadinessDetails: vi.fn(),
}));
vi.mock("@/app/(dashboard)/hooks/latestRelease/useLatestReleaseInfo", () => ({
  useLatestReleaseInfo: vi.fn(),
}));

import { useHealthReadinessDetails } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";
import { useLatestReleaseInfo } from "@/app/(dashboard)/hooks/latestRelease/useLatestReleaseInfo";

const RELEASE: LatestReleaseInfo = {
  version: "1.103.0",
  new_features: 12,
  bug_fixes: 30,
  other_updates: 8,
  release_url: "https://docs.litellm.ai/release_notes/v1.103.0/v1-103-0",
};

const bell = (unread: number) =>
  screen.getByRole("button", { name: unread === 0 ? /^notifications$/i : `Notifications, ${unread} unread` });

const announcement = (title: string) => screen.getByRole("article", { name: title });

describe("NotificationsBellView", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("lists the new version notice above Auto Router when the proxy is behind the latest release", async () => {
    const user = userEvent.setup();
    renderWithProviders(<NotificationsBellView currentVersion="1.102.0" latestRelease={RELEASE} />);
    await user.click(bell(2));

    const titles = screen.getAllByRole("article").map((item) => item.getAttribute("aria-label"));
    expect(titles).toEqual(["LiteLLM v1.103.0 is available", "LiteLLM Auto Router"]);
    const upgrade = announcement("LiteLLM v1.103.0 is available");
    expect(upgrade).toHaveTextContent("12 new features, 30 fixes, and 8 other updates");
    expect(upgrade).toHaveTextContent("Your current version is v1.102.0");
    const notes = screen.getByRole("link", { name: /^view release notes$/i });
    expect(notes).toHaveAttribute("href", RELEASE.release_url);
    expect(notes).toHaveAttribute("rel", "noopener noreferrer");
    const docs = screen.getByRole("link", { name: /^read the docs$/i });
    expect(docs).toHaveAttribute("href", AUTO_ROUTER_DOCS_URL);
    expect(docs).toHaveAttribute("target", "_blank");
  });

  it("uses singular wording for counts of one", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <NotificationsBellView
        currentVersion="1.102.0"
        latestRelease={{ ...RELEASE, new_features: 1, bug_fixes: 1, other_updates: 1 }}
      />,
    );
    await user.click(bell(2));
    expect(announcement("LiteLLM v1.103.0 is available")).toHaveTextContent("1 new feature, 1 fix, and 1 other update");
  });

  it("leaves the counts out when the release notes yielded none", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <NotificationsBellView
        currentVersion="1.102.0"
        latestRelease={{ ...RELEASE, new_features: 0, bug_fixes: 0, other_updates: 0 }}
      />,
    );
    await user.click(bell(2));
    const upgrade = announcement("LiteLLM v1.103.0 is available");
    expect(upgrade).toHaveTextContent("Your current version is v1.102.0");
    expect(upgrade).not.toHaveTextContent("0 new features");
    expect(upgrade).not.toHaveTextContent("other update");
  });

  it.each([
    ["the running version is unknown", undefined, RELEASE],
    ["the latest release is unavailable", "1.102.0", null],
    ["the proxy is up to date", "1.103.0", RELEASE],
    ["the proxy is ahead of the latest release", "1.104.0-dev.1", RELEASE],
  ])("lists only Auto Router when %s", async (_case, currentVersion, latestRelease) => {
    const user = userEvent.setup();
    renderWithProviders(<NotificationsBellView currentVersion={currentVersion} latestRelease={latestRelease} />);
    await user.click(bell(1));
    expect(screen.getAllByRole("article").map((item) => item.getAttribute("aria-label"))).toEqual([
      "LiteLLM Auto Router",
    ]);
  });

  it("marks each announcement read on its own and clears the unread dot once all are read", async () => {
    const user = userEvent.setup();
    renderWithProviders(<NotificationsBellView currentVersion="1.102.0" latestRelease={RELEASE} />);
    await user.click(bell(2));
    await user.click(screen.getAllByRole("button", { name: /^mark as read$/i })[0]);

    expect(bell(1).querySelector("[data-slot=badge]")).not.toBeNull();
    expect(screen.getAllByRole("button", { name: /^mark as read$/i })).toHaveLength(1);
    expect(announcement("LiteLLM Auto Router")).toContainElement(
      screen.getByRole("button", { name: /^mark as read$/i }),
    );
    await user.click(screen.getByRole("button", { name: /^mark as read$/i }));

    expect(bell(0).querySelector("[data-slot=badge]")).toBeNull();
    expect(screen.getAllByRole("article")).toHaveLength(2);
    expect(screen.queryByRole("button", { name: /^mark as read$/i })).not.toBeInTheDocument();
  });

  it("keeps the new version notice read across remounts for the same release", async () => {
    const user = userEvent.setup();
    const first = renderWithProviders(<NotificationsBellView currentVersion="1.102.0" latestRelease={RELEASE} />);
    await user.click(bell(2));
    await user.click(screen.getAllByRole("button", { name: /^mark as read$/i })[0]);
    first.unmount();

    renderWithProviders(<NotificationsBellView currentVersion="1.102.0" latestRelease={RELEASE} />);
    expect(bell(1)).toBeInTheDocument();
  });

  it("shows a newer release as unread after an older release was read", async () => {
    const user = userEvent.setup();
    const { rerender } = renderWithProviders(
      <NotificationsBellView currentVersion="1.102.0" latestRelease={RELEASE} />,
    );
    await user.click(bell(2));
    await user.click(screen.getAllByRole("button", { name: /^mark as read$/i })[0]);
    expect(bell(1)).toBeInTheDocument();

    rerender(<NotificationsBellView currentVersion="1.102.0" latestRelease={{ ...RELEASE, version: "1.104.0" }} />);
    expect(bell(2)).toBeInTheDocument();
    expect(announcement("LiteLLM v1.104.0 is available")).toContainElement(
      screen.getAllByRole("button", { name: /^mark as read$/i })[0],
    );
  });

  it("treats a release whose upgrade banner was dismissed as already read", () => {
    localStorage.setItem("litellm:upgradeBannerDismissed:1.103.0", "true");
    renderWithProviders(<NotificationsBellView currentVersion="1.102.0" latestRelease={RELEASE} />);
    expect(bell(1)).toBeInTheDocument();
  });

  it("keeps Auto Router read when it was read before", async () => {
    localStorage.setItem("litellmHideAutoRouterAnnouncement", "true");
    const user = userEvent.setup();
    renderWithProviders(<NotificationsBellView currentVersion="1.103.0" latestRelease={RELEASE} />);
    await user.click(bell(0));
    expect(announcement("LiteLLM Auto Router")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^mark as read$/i })).not.toBeInTheDocument();
  });

  it("syncs sibling bells when one marks an announcement read", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <>
        <NotificationsBellView currentVersion="1.102.0" latestRelease={RELEASE} />
        <NotificationsBellView currentVersion="1.102.0" latestRelease={RELEASE} />
      </>,
    );
    const [bellA] = screen.getAllByRole("button", { name: "Notifications, 2 unread" });
    await user.click(bellA);
    await user.click(screen.getAllByRole("button", { name: /^mark as read$/i })[0]);

    expect(screen.getAllByRole("button", { name: "Notifications, 1 unread" })).toHaveLength(2);
  });
});

describe("NotificationsBell", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("builds the new version notice from the proxy version and the latest release", async () => {
    vi.mocked(useHealthReadinessDetails).mockReturnValue({
      data: { status: "healthy", litellm_version: "1.102.0" },
    } as ReturnType<typeof useHealthReadinessDetails>);
    vi.mocked(useLatestReleaseInfo).mockReturnValue({ data: RELEASE } as ReturnType<typeof useLatestReleaseInfo>);
    const user = userEvent.setup();
    renderWithProviders(<NotificationsBell accessToken="token" />);
    await user.click(bell(2));
    expect(announcement("LiteLLM v1.103.0 is available")).toHaveTextContent("Your current version is v1.102.0");
  });
});

const DEPRECATION = {
  id: "sonnet:2026-11-30:imminent",
  usageModel: "anthropic/claude-sonnet-4-5",
  requests: 18432,
  lastUsedDate: "2026-11-04",
  lastDayRequests: 123,
  model: {
    model_name: "support",
    litellm_model: "anthropic/claude-sonnet-4-5",
    deprecation_date: "2026-11-30",
    days_until_deprecation: 26,
    status: "imminent" as const,
    successor_model: "recommended-model",
  },
};

describe("model deprecation notifications", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());
  it("links a concise deprecation summary to its detail and the unified inbox", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <NotificationsBellView currentVersion={null} latestRelease={null} deprecationAlerts={[DEPRECATION]} />,
    );
    await user.click(screen.getByRole("button", { name: /alerts need attention/ }));
    expect(screen.getByRole("link", { name: /anthropic\/claude-sonnet-4-5 retires/ })).toHaveAttribute(
      "href",
      "/ui/alerts?type=model-deprecations&model=anthropic%2Fclaude-sonnet-4-5",
    );
    expect(screen.getByRole("link", { name: "See all" })).toHaveAttribute("href", "/ui/alerts");
    expect(screen.queryByRole("button", { name: "Mark resolved" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Dismiss", exact: true })).not.toBeInTheDocument();
  });
  it("should clear the warning when the alert has a shared resolved acknowledgement", () => {
    const state = {
      model: DEPRECATION.usageModel,
      deprecation_date: "2026-11-30",
      status: "resolved" as const,
      last_seen_date: "2026-11-04",
      last_seen_requests: 123,
    };
    renderWithProviders(
      <NotificationsBellView
        currentVersion={null}
        latestRelease={null}
        deprecationAlerts={[{ ...DEPRECATION, state }]}
      />,
    );
    expect(bell(1)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /alerts need attention/ })).not.toBeInTheDocument();
  });
  it("should reopen a resolved alert when more requests arrive on the last-seen day", () => {
    const state = {
      model: DEPRECATION.usageModel,
      deprecation_date: "2026-11-30",
      status: "resolved" as const,
      last_seen_date: "2026-11-04",
      last_seen_requests: 122,
    };
    renderWithProviders(
      <NotificationsBellView
        currentVersion={null}
        latestRelease={null}
        deprecationAlerts={[{ ...DEPRECATION, state }]}
      />,
    );
    expect(screen.getByRole("button", { name: /alerts need attention/ })).toBeInTheDocument();
  });
});

const PERFORMANCE: SavedAlert = {
  id: "9007199254740993",
  type: "spend",
  resource_type: "team",
  resource_id: "example-team",
  resource_name: "Research",
  status: "open",
  window_start: "2026-11-10T00:00:00Z",
  window_end: "2026-11-11T00:00:00Z",
  created_at: "2026-11-11T03:00:00Z",
  updated_at: "2026-11-11T03:00:00Z",
  evidence: { value: 1200, baseline: 200, unit: "usd", sample_size: 1000 },
};

describe("combined alert notifications", () => {
  it("shows at most five open alerts across both sources with detail links", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <NotificationsBellView
        currentVersion={null}
        latestRelease={null}
        deprecationAlerts={[DEPRECATION]}
        performanceAlerts={[
          { ...PERFORMANCE, id: "resolved", status: "resolved" },
          { ...PERFORMANCE, id: "dismissed", status: "dismissed" },
          PERFORMANCE,
          ...["2", "3", "4", "5"].map((id) => ({ ...PERFORMANCE, id, resource_name: `Team ${id}` })),
        ]}
      />,
    );
    await user.click(screen.getByRole("button", { name: /alerts need attention/ }));
    const alerts = within(screen.getByRole("region", { name: "Alerts" }));
    expect(alerts.getAllByRole("listitem")).toHaveLength(5);
    expect(alerts.getByRole("link", { name: /Spend on Research/ })).toHaveAttribute(
      "href",
      `/ui/alerts?alert=${PERFORMANCE.id}`,
    );
    expect(alerts.getByRole("link", { name: /Spend on Team 4/ })).toBeInTheDocument();
    expect(alerts.queryByRole("link", { name: /Spend on Team 5/ })).not.toBeInTheDocument();
    expect(alerts.getByRole("link", { name: "See all" })).toHaveAttribute("href", "/ui/alerts");
    expect(alerts.queryByRole("button")).not.toBeInTheDocument();
  });
});
