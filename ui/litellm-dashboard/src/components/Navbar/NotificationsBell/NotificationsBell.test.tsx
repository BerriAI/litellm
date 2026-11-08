import { renderWithProviders, screen } from "../../../../tests/test-utils";
import { AUTO_ROUTER_DOCS_URL, NotificationsBell, NotificationsBellView } from "./NotificationsBell";
import type { LatestReleaseInfo } from "@/app/(dashboard)/hooks/latestRelease/useLatestReleaseInfo";
import { SPEND_ISSUE, RETIREMENT_ISSUE } from "@/components/Issues/issueFixtures";
import React from "react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails", () => ({
  useHealthReadinessDetails: vi.fn(),
}));
vi.mock("@/app/(dashboard)/hooks/latestRelease/useLatestReleaseInfo", () => ({
  useLatestReleaseInfo: vi.fn(),
}));
vi.mock("@/components/Issues/issueApi", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/Issues/issueApi")>()),
  useIssues: () => ({ data: undefined, enabled: false }),
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
    await user.click(screen.getByRole("button", { name: /What's new/ }));

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
    await user.click(screen.getByRole("button", { name: /What's new/ }));
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
    await user.click(screen.getByRole("button", { name: /What's new/ }));
    expect(screen.getAllByRole("article").map((item) => item.getAttribute("aria-label"))).toEqual([
      "LiteLLM Auto Router",
    ]);
  });

  it("marks each announcement read on its own and clears the unread dot once all are read", async () => {
    const user = userEvent.setup();
    renderWithProviders(<NotificationsBellView currentVersion="1.102.0" latestRelease={RELEASE} />);
    await user.click(bell(2));
    await user.click(screen.getByRole("button", { name: /What's new/ }));
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
    await user.click(screen.getByRole("button", { name: /What's new/ }));
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
    await user.click(screen.getByRole("button", { name: /What's new/ }));
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
    await user.click(screen.getByRole("button", { name: /What's new/ }));
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
    await user.click(screen.getByRole("button", { name: /What's new/ }));
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
    await user.click(screen.getByRole("button", { name: /What's new/ }));
    expect(announcement("LiteLLM v1.103.0 is available")).toHaveTextContent("Your current version is v1.102.0");
  });
});

const ISSUES = { data: [SPEND_ISSUE, RETIREMENT_ISSUE], total: 2, limit: 3, offset: 0 };

describe("issue notifications", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  it.each([
    ["unavailable", undefined],
    ["empty", { data: [], total: 0, limit: 3, offset: 0 }],
    ["active", ISSUES],
  ] as const)("keeps updates collapsed when alerts are %s", async (_state, issues) => {
    const user = userEvent.setup();
    renderWithProviders(<NotificationsBellView currentVersion={null} latestRelease={null} issues={issues} />);
    await user.click(screen.getByRole("button", { name: /^Notifications/ }));
    expect(screen.getByRole("button", { name: /What's new/ })).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("article", { name: "LiteLLM Auto Router" })).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Docs", exact: true })).toBeVisible();
    await user.click(screen.getByRole("button", { name: /What's new/ }));
    const update = announcement("LiteLLM Auto Router");
    expect(screen.getByLabelText("Notification content")).toContainElement(update);
  });

  it("opens issue-owned investigations and keeps resources outside the scrolling content", async () => {
    const user = userEvent.setup();
    renderWithProviders(<NotificationsBellView currentVersion={null} latestRelease={null} issues={ISSUES} />);
    await user.click(screen.getByRole("button", { name: "Notifications, 2 alerts need attention" }));
    const destination = screen.getByRole("link", { name: SPEND_ISSUE.title });
    expect(destination).toHaveAttribute("href", "/ui/alerts?alert=issue-checkout-spend");
    expect(screen.getByRole("article", { name: SPEND_ISSUE.title })).toHaveTextContent(
      "$600.00 on Nov 4, 2026 UTC, compared with the usual $30.00.",
    );
    expect(screen.getByRole("link", { name: "View all 2" })).toHaveAttribute("href", "/ui/alerts");
    const content = screen.getByLabelText("Notification content");
    expect(content).toContainElement(destination);
    expect(content).not.toContainElement(screen.getByRole("link", { name: "Docs", exact: true }));
    expect(content).not.toContainElement(screen.getByRole("link", { name: "Blog", exact: true }));
    expect(screen.queryByRole("article", { name: "LiteLLM Auto Router" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /What's new/ }));
    expect(announcement("LiteLLM Auto Router")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /^mark as read$/i }));
    expect(screen.getByRole("button", { name: "Notifications, 2 alerts need attention" })).toBeInTheDocument();
  });

  it("preserves retirement recommendation provenance alongside performance issues", async () => {
    const user = userEvent.setup();
    renderWithProviders(<NotificationsBellView currentVersion={null} latestRelease={null} issues={ISSUES} />);
    await user.click(screen.getByRole("button", { name: "Notifications, 2 alerts need attention" }));
    expect(screen.getByRole("link", { name: RETIREMENT_ISSUE.title })).toHaveAttribute(
      "href",
      "/ui/alerts?alert=issue-model-retirement",
    );
    expect(screen.getByRole("link", { name: "LiteLLM model cost map" })).toHaveAttribute(
      "href",
      "https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json",
    );
    const notice = screen.getByRole("link", { name: /Provider deprecation notice/ });
    expect(notice).toHaveAttribute("href", "https://provider.example/model-retirements");
    expect(notice).toHaveAttribute("rel", "noopener noreferrer");
    expect(screen.getByRole("article", { name: RETIREMENT_ISSUE.title })).toContainElement(notice);
  });
});
