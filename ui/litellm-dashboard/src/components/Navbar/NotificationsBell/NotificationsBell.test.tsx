import { renderWithProviders, screen } from "../../../../tests/test-utils";
import { AUTO_ROUTER_DOCS_URL, NotificationsBell, NotificationsBellView } from "./NotificationsBell";
import type { LatestReleaseInfo } from "@/app/(dashboard)/hooks/latestRelease/useLatestReleaseInfo";
import React from "react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

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
