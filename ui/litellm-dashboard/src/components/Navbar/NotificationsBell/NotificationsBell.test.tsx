import { renderWithProviders, screen } from "../../../../tests/test-utils";
import { AUTO_ROUTER_DOCS_URL, NotificationsBell, NotificationsBellView } from "./NotificationsBell";
import type { LatestReleaseInfo } from "@/app/(dashboard)/hooks/latestRelease/useLatestReleaseInfo";
import type { AlertGroupPage, UsageAlert } from "@/components/UsageAlerts/usageAlertApi";
import React from "react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/components/ModelDeprecationAlerts/useDeprecationAlerts", () => ({
  useDeprecationAlerts: () => ({
    data: [],
    scope: "test",
    range: { start_date: "2026-10-06", end_date: "2026-11-04" },
  }),
}));

vi.mock("@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails", () => ({
  useHealthReadinessDetails: vi.fn(),
}));
vi.mock("@/app/(dashboard)/hooks/latestRelease/useLatestReleaseInfo", () => ({
  useLatestReleaseInfo: vi.fn(),
}));
vi.mock("@/components/UsageAlerts/usageAlertApi", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/UsageAlerts/usageAlertApi")>()),
  useUsageAlertGroups: () => ({ data: undefined }),
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

const ACTIVITY_PATH =
  "/api-keys?key=reporting-test-id&detail_tab=activity&activity_from=2026-10-28&activity_to=2026-11-04&alert=incident-test";
const INCIDENT: UsageAlert = {
  id: "incident-test",
  rule_id: "rule-test",
  rule_name: "Daily cost",
  scope_type: "key",
  scope_id: "reporting-test-id",
  metric: "spend",
  direction: "increase",
  status: "open",
  day: "2026-11-04",
  observed_value: 120,
  baseline_value: 30,
  baseline_start: "2026-10-28",
  baseline_end: "2026-11-03",
  threshold: 90,
  first_seen: "2026-11-05T01:00:00Z",
  last_seen: "2026-11-05T01:00:00Z",
  notification_attempts: 0,
  notified_channels: [],
  activity_path: ACTIVITY_PATH,
};
const GROUPS: AlertGroupPage = {
  data: [
    {
      id: "group-test",
      scope_type: "key",
      scope_id: "reporting-test-id",
      scope_label: "Checkout key",
      day: "2026-11-04",
      incident_count: 4,
      activity_path: ACTIVITY_PATH,
      metrics: (["spend", "requests", "errors", "latency"] as const).map((metric) => ({
        metric,
        incident_count: 1,
        representative: { ...INCIDENT, id: `incident-${metric}`, metric },
      })),
    },
  ],
  total: 1,
  limit: 3,
  offset: 0,
};

describe("grouped performance notifications", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  it("shows one resource link for its four signals and keeps resources outside the scroll area", async () => {
    const user = userEvent.setup();
    renderWithProviders(<NotificationsBellView currentVersion={null} latestRelease={null} usageAlertGroups={GROUPS} />);
    await user.click(screen.getByRole("button", { name: "Notifications, 1 alert group needs attention" }));
    const destination = screen.getByRole("link", { name: /Checkout key/ });
    expect(destination).toHaveAttribute("href", `/ui${ACTIVITY_PATH}`);
    expect(destination).toHaveTextContent("Requests ↑ 4×");
    expect(destination).toHaveTextContent("Errors ↑ 4×");
    expect(destination).toHaveTextContent("2026-11-04 UTC");
    expect(screen.getByRole("link", { name: "View all 1" })).toHaveAttribute("href", "/ui/alerts?type=usage");
    const content = screen.getByLabelText("Notification content");
    expect(content).toContainElement(destination);
    expect(content).not.toContainElement(screen.getByRole("link", { name: "Docs", exact: true }));
    expect(content).not.toContainElement(screen.getByRole("link", { name: "Blog", exact: true }));
    expect(screen.queryByRole("article", { name: "LiteLLM Auto Router" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /What’s new/ }));
    expect(announcement("LiteLLM Auto Router")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /^mark as read$/i }));
    expect(screen.getByRole("button", { name: "Notifications, 1 alert group needs attention" })).toBeInTheDocument();
  });

  it("preserves deprecation provenance and destinations alongside Performance groups", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <NotificationsBellView
        currentVersion={null}
        latestRelease={null}
        usageAlertGroups={GROUPS}
        deprecationAlerts={[
          {
            ...DEPRECATION,
            model: {
              ...DEPRECATION.model,
              successor_model_source: "model_cost_map",
              deprecation_source: "https://provider.example/model-retirements",
            },
          },
        ]}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Notifications, 2 alert groups need attention" }));
    expect(screen.getByRole("link", { name: /Checkout key/ })).toHaveAttribute("href", `/ui${ACTIVITY_PATH}`);
    expect(screen.getByRole("link", { name: "Review migration" })).toHaveAttribute(
      "href",
      "/ui/alerts?type=model-deprecations&model=anthropic%2Fclaude-sonnet-4-5",
    );
    expect(screen.getByRole("link", { name: "LiteLLM model cost map" })).toHaveAttribute(
      "href",
      "https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json",
    );
    const notice = screen.getByRole("link", { name: /Provider deprecation notice/ });
    expect(notice).toHaveAttribute("href", "https://provider.example/model-retirements");
    expect(notice).toHaveAttribute("target", "_blank");
    expect(notice).toHaveAttribute("rel", "noopener noreferrer");
    expect(screen.getByRole("article", { name: /retires in 26 days/ })).toContainElement(notice);
  });
});

describe("model deprecation notifications", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());
  it("should show last seen and link to the migration workspace and open alerts inbox", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <NotificationsBellView currentVersion={null} latestRelease={null} deprecationAlerts={[DEPRECATION]} />,
    );
    await user.click(screen.getByRole("button", { name: /1 model deprecation alert/ }));
    expect(screen.getByRole("article", { name: /retires in 26 days/ })).toHaveTextContent("Last seen 2026-11-04 UTC");
    expect(screen.getByRole("link", { name: "Review migration" })).toHaveAttribute(
      "href",
      "/ui/alerts?type=model-deprecations&model=anthropic%2Fclaude-sonnet-4-5",
    );
    expect(screen.getByRole("link", { name: "View all 1" })).toHaveAttribute(
      "href",
      "/ui/alerts?type=model-deprecations",
    );
    expect(screen.getByText("recommended-model")).toBeInTheDocument();
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
    expect(screen.queryByRole("button", { name: /model deprecation alert/ })).not.toBeInTheDocument();
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
    expect(screen.getByRole("button", { name: /1 model deprecation alert/ })).toBeInTheDocument();
  });
});
