import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen } from "../../../../tests/test-utils";
import { CommunityEngagementButtons } from "./CommunityEngagementButtons";

let mockUseDisableShowPromptsImpl = () => false;

vi.mock("@/app/(dashboard)/hooks/useDisableShowPrompts", () => ({
  useDisableShowPrompts: () => mockUseDisableShowPromptsImpl(),
}));

describe("CommunityEngagementButtons", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseDisableShowPromptsImpl = () => false;
  });

  it("should render", () => {
    renderWithProviders(<CommunityEngagementButtons />);
    expect(screen.getByRole("link", { name: /join slack/i })).toBeInTheDocument();
  });

  it("should render Join Slack button with correct link", () => {
    renderWithProviders(<CommunityEngagementButtons />);

    const joinSlackLink = screen.getByRole("link", { name: /join slack/i });
    expect(joinSlackLink).toBeInTheDocument();
    expect(joinSlackLink).toHaveAttribute("href", "https://www.litellm.ai/support");
    expect(joinSlackLink).toHaveAttribute("target", "_blank");
    expect(joinSlackLink).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("should render Star us on GitHub button with correct link", () => {
    renderWithProviders(<CommunityEngagementButtons />);

    const starOnGithubLink = screen.getByRole("link", { name: /star us on github/i });
    expect(starOnGithubLink).toBeInTheDocument();
    expect(starOnGithubLink).toHaveAttribute("href", "https://github.com/BerriAI/litellm");
    expect(starOnGithubLink).toHaveAttribute("target", "_blank");
    expect(starOnGithubLink).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("should not render buttons when prompts are disabled", () => {
    mockUseDisableShowPromptsImpl = () => true;

    renderWithProviders(<CommunityEngagementButtons />);

    expect(screen.queryByRole("link", { name: /join slack/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /star us on github/i })).not.toBeInTheDocument();
  });
});
