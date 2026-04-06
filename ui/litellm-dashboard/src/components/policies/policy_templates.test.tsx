import React from "react";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "../networking";
import PolicyTemplates from "./policy_templates";

vi.mock("../networking");

vi.mock("@heroicons/react/outline", () => ({
  ShieldCheckIcon: function ShieldCheckIcon() { return null; },
  ShieldExclamationIcon: function ShieldExclamationIcon() { return null; },
  BeakerIcon: function BeakerIcon() { return null; },
  CurrencyDollarIcon: function CurrencyDollarIcon() { return null; },
  CheckCircleIcon: function CheckCircleIcon() { return null; },
}));

const makeTemplate = (overrides: any = {}) => ({
  id: "tpl-1",
  title: "Test Template",
  description: "A test template",
  icon: "ShieldCheckIcon",
  iconColor: "text-green-500",
  iconBg: "bg-green-50",
  guardrails: ["guardrail-a"],
  tags: [],
  complexity: "Low" as const,
  ...overrides,
});

const defaultProps = {
  onUseTemplate: vi.fn(),
  onOpenAiSuggestion: vi.fn(),
  accessToken: "test-token",
};

describe("PolicyTemplates", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the section header after loading", async () => {
    vi.mocked(networking.getPolicyTemplates).mockResolvedValue([]);
    renderWithProviders(<PolicyTemplates {...defaultProps} />);
    expect(await screen.findByText("Policy Templates")).toBeInTheDocument();
  });

  it("should not show the template grid while fetching", () => {
    vi.mocked(networking.getPolicyTemplates).mockReturnValue(new Promise(() => {}));
    renderWithProviders(<PolicyTemplates {...defaultProps} />);
    expect(screen.queryByText("Policy Templates")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /use ai to find templates/i })).not.toBeInTheDocument();
  });

  it("should render a card for each fetched template", async () => {
    const templates = [
      makeTemplate({ title: "Template Alpha" }),
      makeTemplate({ id: "tpl-2", title: "Template Beta" }),
    ];
    vi.mocked(networking.getPolicyTemplates).mockResolvedValue(templates);
    renderWithProviders(<PolicyTemplates {...defaultProps} />);
    expect(await screen.findByText("Template Alpha")).toBeInTheDocument();
    expect(screen.getByText("Template Beta")).toBeInTheDocument();
  });

  it("should call onTemplatesLoaded with the fetched templates after loading", async () => {
    const templates = [makeTemplate()];
    vi.mocked(networking.getPolicyTemplates).mockResolvedValue(templates);
    const onTemplatesLoaded = vi.fn();
    renderWithProviders(<PolicyTemplates {...defaultProps} onTemplatesLoaded={onTemplatesLoaded} />);
    await waitFor(() => {
      expect(onTemplatesLoaded).toHaveBeenCalledWith(templates);
    });
  });

  it("should call onOpenAiSuggestion when the AI suggestion button is clicked", async () => {
    vi.mocked(networking.getPolicyTemplates).mockResolvedValue([]);
    const user = userEvent.setup();
    renderWithProviders(<PolicyTemplates {...defaultProps} />);
    await screen.findByText("Policy Templates");
    await user.click(screen.getByRole("button", { name: /use ai to find templates/i }));
    expect(defaultProps.onOpenAiSuggestion).toHaveBeenCalled();
  });

  it("should render tag filter checkboxes for unique tags across all templates", async () => {
    const templates = [
      makeTemplate({ tags: ["compliance"] }),
      makeTemplate({ id: "tpl-2", tags: ["compliance", "security"] }),
    ];
    vi.mocked(networking.getPolicyTemplates).mockResolvedValue(templates);
    renderWithProviders(<PolicyTemplates {...defaultProps} />);
    expect(await screen.findByRole("checkbox", { name: /compliance/i })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /security/i })).toBeInTheDocument();
  });

  it("should filter to only matching templates when a tag is selected", async () => {
    const templates = [
      makeTemplate({ id: "tpl-1", title: "Compliance Template", tags: ["compliance"] }),
      makeTemplate({ id: "tpl-2", title: "Security Template", tags: ["security"] }),
    ];
    vi.mocked(networking.getPolicyTemplates).mockResolvedValue(templates);
    const user = userEvent.setup();
    renderWithProviders(<PolicyTemplates {...defaultProps} />);
    await screen.findByText("Compliance Template");
    await user.click(screen.getByRole("checkbox", { name: /compliance/i }));
    expect(screen.getByText("Compliance Template")).toBeInTheDocument();
    expect(screen.queryByText("Security Template")).not.toBeInTheDocument();
  });

  it("should show 'No templates match' when selected tags exclude all templates", async () => {
    const templates = [
      makeTemplate({ id: "tpl-1", title: "Alpha Template", tags: ["alpha"] }),
      makeTemplate({ id: "tpl-2", title: "Beta Template", tags: ["beta"] }),
    ];
    vi.mocked(networking.getPolicyTemplates).mockResolvedValue(templates);
    const user = userEvent.setup();
    renderWithProviders(<PolicyTemplates {...defaultProps} />);
    await screen.findByText("Alpha Template");
    await user.click(screen.getByRole("checkbox", { name: /alpha/i }));
    await user.click(screen.getByRole("checkbox", { name: /beta/i }));
    expect(screen.getByText(/no templates match the selected filters/i)).toBeInTheDocument();
  });

  it("should restore all templates when 'Clear all' is clicked", async () => {
    const templates = [
      makeTemplate({ id: "tpl-1", title: "Alpha Template", tags: ["alpha"] }),
      makeTemplate({ id: "tpl-2", title: "Beta Template", tags: ["beta"] }),
    ];
    vi.mocked(networking.getPolicyTemplates).mockResolvedValue(templates);
    const user = userEvent.setup();
    renderWithProviders(<PolicyTemplates {...defaultProps} />);
    await screen.findByText("Alpha Template");
    await user.click(screen.getByRole("checkbox", { name: /alpha/i }));
    expect(screen.queryByText("Beta Template")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /clear all/i }));
    expect(screen.getByText("Beta Template")).toBeInTheDocument();
  });

  it("should not fetch templates when accessToken is null", () => {
    renderWithProviders(<PolicyTemplates {...defaultProps} accessToken={null} />);
    expect(networking.getPolicyTemplates).not.toHaveBeenCalled();
  });
});
