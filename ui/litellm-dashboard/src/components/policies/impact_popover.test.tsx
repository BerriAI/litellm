import React from "react";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "../networking";
import ImpactPopover from "./impact_popover";
import { PolicyAttachment } from "./types";

vi.mock("../networking");

vi.mock("@heroicons/react/outline", () => ({
  EyeIcon: function EyeIcon() { return null; },
}));

vi.mock("@tremor/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tremor/react")>();
  return {
    ...actual,
    Icon: ({ icon: IconComp, onClick, className }: any) =>
      React.createElement("button", { type: "button", onClick, className }, IconComp?.displayName ?? IconComp?.name ?? "icon"),
  };
});

// Expose the Popover's onOpenChange so tests can trigger it programmatically.
vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<any>();
  return {
    ...actual,
    Popover: ({ children, onOpenChange, content }: any) =>
      React.createElement(
        "div",
        null,
        React.createElement("div", { "data-testid": "popover-content" }, content),
        React.createElement(
          "div",
          {
            role: "button",
            "aria-label": "open-popover",
            onClick: () => onOpenChange?.(true),
          },
          children
        )
      ),
    Tooltip: ({ children }: any) => React.createElement(React.Fragment, null, children),
    Spin: () => React.createElement("span", null, "Loading..."),
    Tag: ({ children }: any) => React.createElement("span", null, children),
  };
});

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
  });

  it("should render", () => {
    renderWithProviders(<ImpactPopover attachment={makeAttachment()} accessToken="tok" />);
    expect(screen.getByRole("button", { name: /open-popover/i })).toBeInTheDocument();
  });

  it("should show 'Click to load' as the initial popover content", () => {
    renderWithProviders(<ImpactPopover attachment={makeAttachment()} accessToken="tok" />);
    expect(screen.getByText(/click to load/i)).toBeInTheDocument();
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
    await user.click(screen.getByRole("button", { name: /open-popover/i }));
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
    await user.click(screen.getByRole("button", { name: /open-popover/i }));
    expect(networking.estimateAttachmentImpactCall).not.toHaveBeenCalled();
  });

  it("should show a loading indicator while the impact is being fetched", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.estimateAttachmentImpactCall).mockReturnValue(new Promise(() => {}));
    renderWithProviders(<ImpactPopover attachment={makeAttachment()} accessToken="tok" />);
    await user.click(screen.getByRole("button", { name: /open-popover/i }));
    // Multiple "Loading..." nodes exist (Spin + adjacent text) — assert at least one is present
    expect(screen.queryAllByText(/loading/i).length).toBeGreaterThan(0);
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
    await user.click(screen.getByRole("button", { name: /open-popover/i }));
    expect(await screen.findByText(/global scope.*affects all keys and teams/i)).toBeInTheDocument();
  });

  it("should show key and team counts when impact data is loaded for a specific scope", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.estimateAttachmentImpactCall).mockResolvedValue({
      affected_keys_count: 5,
      affected_teams_count: 2,
      sample_keys: ["sk-abc"],
      sample_teams: ["team-x"],
    });
    renderWithProviders(<ImpactPopover attachment={makeAttachment()} accessToken="tok" />);
    await user.click(screen.getByRole("button", { name: /open-popover/i }));
    expect(await screen.findByText(/5/)).toBeInTheDocument();
    expect(screen.getByText(/2/)).toBeInTheDocument();
  });

  it("should render sample key tags when returned from the API", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.estimateAttachmentImpactCall).mockResolvedValue({
      affected_keys_count: 2,
      affected_teams_count: 0,
      sample_keys: ["sk-key-one", "sk-key-two"],
      sample_teams: [],
    });
    renderWithProviders(<ImpactPopover attachment={makeAttachment()} accessToken="tok" />);
    await user.click(screen.getByRole("button", { name: /open-popover/i }));
    expect(await screen.findByText("sk-key-one")).toBeInTheDocument();
    expect(screen.getByText("sk-key-two")).toBeInTheDocument();
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
    await user.click(screen.getByRole("button", { name: /open-popover/i }));
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
    await user.click(screen.getByRole("button", { name: /open-popover/i }));
    await screen.findByText("sk-abc");
    await user.click(screen.getByRole("button", { name: /open-popover/i }));
    expect(networking.estimateAttachmentImpactCall).toHaveBeenCalledTimes(1);
  });
});
