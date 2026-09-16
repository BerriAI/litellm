import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GuardrailGarden from "./guardrail_garden";
import { ALL_CARDS } from "./guardrail_garden_data";

vi.mock("./guardrail_garden_detail", () => ({
  __esModule: true,
  default: ({ card, onBack }: { card: { name: string }; onBack: () => void }) => (
    <div>
      <span>Detail for {card.name}</span>
      <button onClick={onBack}>Back to garden</button>
    </div>
  ),
}));

const LITELLM_CARDS = ALL_CARDS.filter((c) => c.category === "litellm");
const PARTNER_CARDS = ALL_CARDS.filter((c) => c.category === "partner");

describe("GuardrailGarden", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const renderGarden = () => render(<GuardrailGarden accessToken="test-token" onGuardrailCreated={vi.fn()} />);

  it("should render both sections with their descriptions", () => {
    renderGarden();

    expect(screen.getByText("LiteLLM Content Filter")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Built-in guardrails powered by LiteLLM. Zero latency, no external dependencies, no additional cost.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("Partner Guardrails")).toBeInTheDocument();
    expect(
      screen.getByText("Third-party guardrail integrations from leading AI security providers."),
    ).toBeInTheDocument();
  });

  it("should show a capped set of litellm cards behind a show all toggle", async () => {
    const user = userEvent.setup();
    renderGarden();

    expect(screen.getByText(`Show all (${LITELLM_CARDS.length})`)).toBeInTheDocument();
    expect(screen.getByText(LITELLM_CARDS[0].name)).toBeInTheDocument();
    expect(screen.queryByText(LITELLM_CARDS[LITELLM_CARDS.length - 1].name)).not.toBeInTheDocument();

    await user.click(screen.getByText(`Show all (${LITELLM_CARDS.length})`));

    expect(screen.getByText("Show less")).toBeInTheDocument();
    expect(screen.getByText(LITELLM_CARDS[LITELLM_CARDS.length - 1].name)).toBeInTheDocument();
  });

  it("should always render every partner card", () => {
    renderGarden();

    PARTNER_CARDS.forEach((card) => {
      expect(screen.getByText(card.name)).toBeInTheDocument();
    });
  });

  it("should filter cards by the search query", async () => {
    const user = userEvent.setup();
    renderGarden();

    const target = PARTNER_CARDS[0];
    await user.type(screen.getByPlaceholderText("Search guardrails"), target.name);

    expect(await screen.findByText(target.name)).toBeInTheDocument();
    const otherPartner = PARTNER_CARDS.find((c) => c.name !== target.name);
    if (otherPartner) {
      expect(screen.queryByText(otherPartner.name)).not.toBeInTheDocument();
    }
  });

  it("should show an empty result set for a query that matches nothing", async () => {
    const user = userEvent.setup();
    renderGarden();

    await user.type(screen.getByPlaceholderText("Search guardrails"), "zzzzznotaguardrailzzzzz");

    expect(screen.getByText("Show all (0)")).toBeInTheDocument();
    PARTNER_CARDS.forEach((card) => {
      expect(screen.queryByText(card.name)).not.toBeInTheDocument();
    });
  });

  it("should open the detail view for a clicked card and return to the garden", async () => {
    const user = userEvent.setup();
    renderGarden();

    const target = PARTNER_CARDS[0];
    await user.click(screen.getByText(target.name));

    expect(await screen.findByText(`Detail for ${target.name}`)).toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Search guardrails")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Back to garden" }));

    expect(await screen.findByPlaceholderText("Search guardrails")).toBeInTheDocument();
  });
});
