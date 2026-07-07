import { renderWithProviders, screen, within } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import ComplexityRouterConfig from "./ComplexityRouterConfig";

const mockModelInfo = [
  { model_group: "gpt-4" },
  { model_group: "gpt-3.5-turbo" },
  { model_group: "claude-3-opus" },
] as any[];

const defaultTiers = {
  SIMPLE: "gpt-3.5-turbo",
  MEDIUM: "gpt-3.5-turbo",
  COMPLEX: "gpt-4",
  REASONING: "claude-3-opus",
};

describe("ComplexityRouterConfig", () => {
  it("should render", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultTiers} onChange={vi.fn()} />);
    expect(screen.getByText("Complexity Tier Configuration")).toBeInTheDocument();
  });

  it("should display all four tier labels", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultTiers} onChange={vi.fn()} />);
    expect(screen.getByText("Simple Tier")).toBeInTheDocument();
    expect(screen.getByText("Medium Tier")).toBeInTheDocument();
    expect(screen.getByText("Complex Tier")).toBeInTheDocument();
    expect(screen.getByText("Reasoning Tier")).toBeInTheDocument();
  });

  it("should show example queries for each tier", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultTiers} onChange={vi.fn()} />);
    expect(screen.getByText(/Hello!/)).toBeInTheDocument();
    expect(screen.getByText(/Explain how REST APIs work/)).toBeInTheDocument();
    expect(screen.getByText(/Design a microservices architecture/)).toBeInTheDocument();
    expect(screen.getByText(/Think step by step/)).toBeInTheDocument();
  });

  it("should display the how classification works section", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultTiers} onChange={vi.fn()} />);
    expect(screen.getByText("How Classification Works")).toBeInTheDocument();
  });

  it("should show score thresholds in the classification section", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultTiers} onChange={vi.fn()} />);
    expect(screen.getByText(/Score < 0.15/)).toBeInTheDocument();
    expect(screen.getByText(/Score 0.15 - 0.35/)).toBeInTheDocument();
    expect(screen.getByText(/Score 0.35 - 0.60/)).toBeInTheDocument();
    expect(screen.getByText(/Score > 0.60/)).toBeInTheDocument();
  });

  it("should render the custom technical keywords field", () => {
    renderWithProviders(<ComplexityRouterConfig modelInfo={mockModelInfo} value={defaultTiers} onChange={vi.fn()} />);
    expect(screen.getByText("Custom Technical Keywords")).toBeInTheDocument();
  });

  it("should display existing custom technical keywords as tags", () => {
    renderWithProviders(
      <ComplexityRouterConfig
        modelInfo={mockModelInfo}
        value={defaultTiers}
        onChange={vi.fn()}
        customTechnicalKeywords={["udp", "kafka"]}
        onCustomTechnicalKeywordsChange={vi.fn()}
      />,
    );
    expect(screen.getByText("udp")).toBeInTheDocument();
    expect(screen.getByText("kafka")).toBeInTheDocument();
  });

  it("should call onCustomTechnicalKeywordsChange when a keyword is entered", async () => {
    const user = userEvent.setup();
    const onCustomTechnicalKeywordsChange = vi.fn();
    renderWithProviders(
      <ComplexityRouterConfig
        modelInfo={mockModelInfo}
        value={defaultTiers}
        onChange={vi.fn()}
        customTechnicalKeywords={[]}
        onCustomTechnicalKeywordsChange={onCustomTechnicalKeywordsChange}
      />,
    );
    const keywordsCard = screen.getByText("Custom Technical Keywords").closest(".ant-card") as HTMLElement;
    const input = within(keywordsCard).getByRole("combobox");
    await user.type(input, "udp,");
    expect(onCustomTechnicalKeywordsChange).toHaveBeenCalledWith(["udp"]);
  });
});
