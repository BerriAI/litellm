import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import SkillPermissionsPicker from "./SkillPermissionsPicker";

vi.mock("@/app/(dashboard)/hooks/claudeCodeMarketplaces/useClaudeCodeMarketplaces", () => ({
  useClaudeCodeMarketplaces: vi.fn(),
}));
vi.mock("@/app/(dashboard)/hooks/claudeCodePlugins/useClaudeCodePlugins", () => ({
  useClaudeCodePlugins: vi.fn(),
}));

import { useClaudeCodeMarketplaces } from "@/app/(dashboard)/hooks/claudeCodeMarketplaces/useClaudeCodeMarketplaces";
import { useClaudeCodePlugins } from "@/app/(dashboard)/hooks/claudeCodePlugins/useClaudeCodePlugins";

const mockUseMarketplaces = vi.mocked(useClaudeCodeMarketplaces);
const mockUsePlugins = vi.mocked(useClaudeCodePlugins);

const marketplaceFixture = (overrides: Partial<Record<string, unknown>> = {}) => ({
  id: "mkt-1",
  name: "anthropic-agent-skills",
  display_name: "Anthropic Agent Skills",
  source_type: "github",
  source_ref: "anthropics/agent-skills",
  enabled: true,
  sync_status: "synced",
  plugin_count: 2,
  created_at: "2024-01-01T00:00:00Z",
  updated_at: "2024-01-01T00:00:00Z",
  ...overrides,
});

const otherMarketplaceFixture = marketplaceFixture({
  id: "mkt-2",
  name: "other-marketplace",
  display_name: "Other Marketplace",
  plugin_count: 1,
});

const pluginFixture = (name: string) => ({
  id: name,
  name,
  enabled: false,
  source: { source: "github" as const },
});

describe("SkillPermissionsPicker", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("bulk-adds every skill belonging to a marketplace when its checkbox is checked", async () => {
    mockUseMarketplaces.mockReturnValue({ data: [marketplaceFixture()], isLoading: false } as any);
    mockUsePlugins.mockReturnValue({
      data: [
        pluginFixture("anthropic-agent-skills--document-skills"),
        pluginFixture("anthropic-agent-skills--pdf-skills"),
      ],
      isLoading: false,
    } as any);

    const onChange = vi.fn();
    renderWithProviders(<SkillPermissionsPicker accessToken="tok" value={[]} onChange={onChange} />);

    const marketplaceCheckbox = screen.getByRole("checkbox", { name: /Anthropic Agent Skills/ });
    await userEvent.click(marketplaceCheckbox);

    expect(onChange).toHaveBeenCalledWith(
      expect.arrayContaining(["anthropic-agent-skills--document-skills", "anthropic-agent-skills--pdf-skills"]),
    );
    expect(onChange.mock.calls[0][0]).toHaveLength(2);
  });

  it("removes only that marketplace's skills when its checkbox is unchecked, leaving other marketplaces untouched", async () => {
    mockUseMarketplaces.mockReturnValue({
      data: [marketplaceFixture(), otherMarketplaceFixture],
      isLoading: false,
    } as any);
    mockUsePlugins.mockReturnValue({
      data: [
        pluginFixture("anthropic-agent-skills--document-skills"),
        pluginFixture("anthropic-agent-skills--pdf-skills"),
        pluginFixture("other-marketplace--some-skill"),
      ],
      isLoading: false,
    } as any);

    const onChange = vi.fn();
    renderWithProviders(
      <SkillPermissionsPicker
        accessToken="tok"
        value={[
          "anthropic-agent-skills--document-skills",
          "anthropic-agent-skills--pdf-skills",
          "other-marketplace--some-skill",
        ]}
        onChange={onChange}
      />,
    );

    // Both marketplace checkboxes render fully checked given the value above.
    const firstMarketplaceCheckbox = screen.getByRole("checkbox", { name: /Anthropic Agent Skills/ });
    await userEvent.click(firstMarketplaceCheckbox);

    expect(onChange).toHaveBeenCalledWith(["other-marketplace--some-skill"]);
  });

  it("removes a single individually-unchecked skill without touching its marketplace siblings", async () => {
    mockUseMarketplaces.mockReturnValue({ data: [marketplaceFixture()], isLoading: false } as any);
    mockUsePlugins.mockReturnValue({
      data: [
        pluginFixture("anthropic-agent-skills--document-skills"),
        pluginFixture("anthropic-agent-skills--pdf-skills"),
      ],
      isLoading: false,
    } as any);

    const onChange = vi.fn();
    renderWithProviders(
      <SkillPermissionsPicker
        accessToken="tok"
        value={["anthropic-agent-skills--document-skills", "anthropic-agent-skills--pdf-skills"]}
        onChange={onChange}
      />,
    );

    const skillCheckbox = screen.getByRole("checkbox", { name: /document-skills/ });
    await userEvent.click(skillCheckbox);

    expect(onChange).toHaveBeenCalledWith(["anthropic-agent-skills--pdf-skills"]);
  });

  it("adds a single individually-checked skill without requiring the marketplace checkbox", async () => {
    mockUseMarketplaces.mockReturnValue({ data: [marketplaceFixture()], isLoading: false } as any);
    mockUsePlugins.mockReturnValue({
      data: [
        pluginFixture("anthropic-agent-skills--document-skills"),
        pluginFixture("anthropic-agent-skills--pdf-skills"),
      ],
      isLoading: false,
    } as any);

    const onChange = vi.fn();
    renderWithProviders(<SkillPermissionsPicker accessToken="tok" value={[]} onChange={onChange} />);

    const skillCheckbox = screen.getByRole("checkbox", { name: /pdf-skills/ });
    await userEvent.click(skillCheckbox);

    expect(onChange).toHaveBeenCalledWith(["anthropic-agent-skills--pdf-skills"]);
  });
});
