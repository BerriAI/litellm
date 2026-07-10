import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GuardrailCard from "./guardrail_garden_card";
import type { GuardrailCardInfo } from "./guardrail_garden_data";

vi.mock("@ant-design/icons", () => ({
  CheckCircleFilled: ({ style, ...props }: any) => <span data-testid="check-icon" {...props} />,
}));

const baseCard: GuardrailCardInfo = {
  id: "test-guard",
  name: "Test Guardrail",
  description: "A guardrail for testing purposes",
  category: "litellm",
  logo: "/logos/test.svg",
  tags: ["safety"],
};

describe("GuardrailCard", () => {
  it("should render", () => {
    render(<GuardrailCard card={baseCard} onClick={vi.fn()} />);
    expect(screen.getByText("Test Guardrail")).toBeInTheDocument();
  });

  it("should display the card description", () => {
    render(<GuardrailCard card={baseCard} onClick={vi.fn()} />);
    expect(screen.getByText("A guardrail for testing purposes")).toBeInTheDocument();
  });

  it("should call onClick when the card is clicked", async () => {
    const onClick = vi.fn();
    const user = userEvent.setup();
    render(<GuardrailCard card={baseCard} onClick={onClick} />);

    await user.click(screen.getByText("Test Guardrail"));

    expect(onClick).toHaveBeenCalled();
  });

  it("should show eval badge when card has eval data", () => {
    const cardWithEval: GuardrailCardInfo = {
      ...baseCard,
      eval: { f1: 95, precision: 92, recall: 98, testCases: 500, latency: "10ms" },
    };
    render(<GuardrailCard card={cardWithEval} onClick={vi.fn()} />);
    expect(screen.getByText(/F1: 95%/)).toBeInTheDocument();
    expect(screen.getByText(/500 test cases/)).toBeInTheDocument();
  });

  it("should not show eval badge when card has no eval data", () => {
    render(<GuardrailCard card={baseCard} onClick={vi.fn()} />);
    expect(screen.queryByText(/F1:/)).not.toBeInTheDocument();
  });

  it("should show fallback initial when logo fails to load", () => {
    render(<GuardrailCard card={baseCard} onClick={vi.fn()} />);
    const img = screen.getByRole("presentation");

    act(() => {
      fireEvent.error(img);
    });

    expect(screen.getByText("T")).toBeInTheDocument();
  });

  it("should show fallback initial when logo src is empty", () => {
    const cardNoLogo: GuardrailCardInfo = { ...baseCard, logo: "" };
    render(<GuardrailCard card={cardNoLogo} onClick={vi.fn()} />);
    expect(screen.getByText("T")).toBeInTheDocument();
  });
});
