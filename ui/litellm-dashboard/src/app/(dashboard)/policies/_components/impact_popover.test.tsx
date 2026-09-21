import React from "react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen, waitFor } from "@/../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "@/components/networking";
import ImpactPopover from "./impact_popover";
import { PolicyAttachment } from "@/components/policies/types";

vi.mock("@/components/networking");

vi.mock("@heroicons/react/outline", () => ({
  EyeIcon: function EyeIcon() {
    return null;
  },
}));

const makeAttachment = (overrides: Partial<PolicyAttachment> = {}): PolicyAttachment => ({
  attachment_id: "att-001",
  policy_name: "my-policy",
  scope: null,
  teams: [],
  keys: [],
  models: [],
  tags: [],
  ...overrides,
});

describe("ImpactPopover", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  it("should render", () => {
    renderWithProviders(<ImpactPopover attachment={makeAttachment()} accessToken="tok" />);
    expect(screen.getByRole("button", { name: /view blast radius/i })).toBeInTheDocument();
  });

  it("should call estimateAttachmentImpactCall when the popover is opened", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.estimateAttachmentImpactCall).mockResolvedValue({
      affected_keys_count: 0,
      affected_teams_count: 0,
      sample_keys: [],
      sample_teams: [],
    });
    const attachment = makeAttachment({ policy_name: "rate-limit", teams: ["team-a"] });
    renderWithProviders(<ImpactPopover attachment={attachment} accessToken="my-token" />);
    await user.click(screen.getByRole("button", { name: /view blast radius/i }));
    await waitFor(() => {
      expect(networking.estimateAttachmentImpactCall).toHaveBeenCalledWith("my-token", {
        policy_name: "rate-limit",
        scope: null,
        teams: ["team-a"],
        keys: [],
        models: [],
        tags: [],
      });
    });
  });

  it("should not call the API when accessToken is null", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ImpactPopover attachment={makeAttachment()} accessToken={null} />);
    await user.click(screen.getByRole("button", { name: /view blast radius/i }));
    expect(networking.estimateAttachmentImpactCall).not.toHaveBeenCalled();
    expect(screen.getByText(/click to load/i)).toBeInTheDocument();
  });

  it("should show a loading indicator while the impact is being fetched", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.estimateAttachmentImpactCall).mockReturnValue(new Promise(() => {}));
    renderWithProviders(<ImpactPopover attachment={makeAttachment()} accessToken="tok" />);
    await user.click(screen.getByRole("button", { name: /view blast radius/i }));
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it("should show a global scope warning when affected_keys_count is -1", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.estimateAttachmentImpactCall).mockResolvedValue({
      affected_keys_count: -1,
      affected_teams_count: -1,
      sample_keys: [],
      sample_teams: [],
    });
    renderWithProviders(<ImpactPopover attachment={makeAttachment()} accessToken="tok" />);
    await user.click(screen.getByRole("button", { name: /view blast radius/i }));
    expect(await screen.findByText(/global scope.*affects all keys and teams/i)).toBeInTheDocument();
  });

  it("should use singular count labels for one affected key and team", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.estimateAttachmentImpactCall).mockResolvedValue({
      affected_keys_count: 1,
      affected_teams_count: 1,
      sample_keys: ["sk-abc"],
      sample_teams: ["team-x"],
    });
    renderWithProviders(<ImpactPopover attachment={makeAttachment()} accessToken="tok" />);
    await user.click(screen.getByRole("button", { name: /view blast radius/i }));
    expect(
      await screen.findByText((_, element) => element?.textContent === "1 key, 1 team affected"),
    ).toBeInTheDocument();
  });

  it("should render sample keys and teams returned by the API", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.estimateAttachmentImpactCall).mockResolvedValue({
      affected_keys_count: 2,
      affected_teams_count: 1,
      sample_keys: ["sk-key-one", "sk-key-two"],
      sample_teams: ["team-one"],
    });
    renderWithProviders(<ImpactPopover attachment={makeAttachment()} accessToken="tok" />);
    await user.click(screen.getByRole("button", { name: /view blast radius/i }));
    expect(await screen.findByText("sk-key-one")).toBeInTheDocument();
    expect(screen.getByText("sk-key-two")).toBeInTheDocument();
    expect(screen.getByText("team-one")).toBeInTheDocument();
  });

  it("should show 'No keys or teams currently affected' when both counts are 0", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.estimateAttachmentImpactCall).mockResolvedValue({
      affected_keys_count: 0,
      affected_teams_count: 0,
      sample_keys: [],
      sample_teams: [],
    });
    renderWithProviders(<ImpactPopover attachment={makeAttachment()} accessToken="tok" />);
    await user.click(screen.getByRole("button", { name: /view blast radius/i }));
    expect(await screen.findByText(/no keys or teams currently affected/i)).toBeInTheDocument();
  });

  it("should not call the API a second time when the popover is already loaded", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.estimateAttachmentImpactCall).mockResolvedValue({
      affected_keys_count: 1,
      affected_teams_count: 0,
      sample_keys: ["sk-abc"],
      sample_teams: [],
    });
    renderWithProviders(<ImpactPopover attachment={makeAttachment()} accessToken="tok" />);
    const trigger = screen.getByRole("button", { name: /view blast radius/i });
    await user.click(trigger);
    await screen.findByText("sk-abc");
    await user.click(trigger);
    await user.click(trigger);
    expect(networking.estimateAttachmentImpactCall).toHaveBeenCalledTimes(1);
  });

  it("should retry loading after a failed request", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.estimateAttachmentImpactCall)
      .mockRejectedValueOnce(new Error("network unavailable"))
      .mockResolvedValueOnce({
        affected_keys_count: 1,
        affected_teams_count: 0,
        sample_keys: ["sk-recovered"],
        sample_teams: [],
      });
    renderWithProviders(<ImpactPopover attachment={makeAttachment()} accessToken="tok" />);

    const trigger = screen.getByRole("button", { name: /view blast radius/i });
    await user.click(trigger);
    expect(await screen.findByText(/click to load/i)).toBeInTheDocument();
    await user.click(trigger);
    await user.click(trigger);

    expect(await screen.findByText("sk-recovered")).toBeInTheDocument();
    expect(console.error).toHaveBeenCalledWith("Failed to load impact:", expect.any(Error));
  });
});
