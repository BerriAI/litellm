import { render, renderWithProviders, screen, waitFor } from "@/../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { describe, expect, it, vi } from "vitest";
import GuardrailDetailView from "./guardrail_garden_detail";
import type { GuardrailCardInfo } from "./guardrail_garden_data";

vi.mock("./add_guardrail_form", () => ({ default: () => null }));

const makeCard = (overrides: Partial<GuardrailCardInfo> = {}): GuardrailCardInfo => ({
  id: "bedrock",
  name: "Bedrock Guardrail",
  description: "AWS Bedrock Guardrails for content filtering.",
  category: "partner",
  logo: "/_next/static/media/bedrock.svg",
  tags: ["AWS"],
  ...overrides,
});

const evalCardOverrides: Partial<GuardrailCardInfo> = {
  id: "cf_denied_financial",
  name: "Denied Financial Advice",
  category: "litellm",
  eval: { f1: 97, precision: 98, recall: 96, testCases: 207, latency: "<0.1ms" },
};
const cardWithEval = makeCard(evalCardOverrides);

const renderDetail = (card: GuardrailCardInfo, searchParams?: string, onUrlUpdate?: OnUrlUpdateFunction) =>
  renderWithProviders(
    <GuardrailDetailView card={card} onBack={vi.fn()} accessToken={null} onGuardrailCreated={vi.fn()} />,
    { searchParams, onUrlUpdate },
  );

const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0];

describe("GuardrailDetailView logo", () => {
  it("renders the card logo through the shared Logo component with the bundled src", () => {
    renderDetail(makeCard());
    expect(screen.getByAltText("Bedrock Guardrail logo")).toHaveAttribute("src", "/_next/static/media/bedrock.svg");
  });

  it("falls back to a letter avatar when the card has no logo", () => {
    renderDetail(makeCard({ logo: "" }));
    expect(screen.queryByAltText("Bedrock Guardrail logo")).not.toBeInTheDocument();
    expect(screen.getByText("B")).toBeInTheDocument();
  });
});

describe("GuardrailDetailView tabs", () => {
  it("shows the overview when the URL names no tab", () => {
    renderDetail(cardWithEval);
    expect(screen.getByText("Guardrail Details")).toBeInTheDocument();
    expect(screen.queryByText("F1 Score")).not.toBeInTheDocument();
  });

  it("shows the eval results when ?garden_tab=eval", () => {
    renderDetail(cardWithEval, "?garden_tab=eval");
    expect(screen.getByText("F1 Score")).toBeInTheDocument();
    expect(screen.getByText("97%")).toBeInTheDocument();
    expect(screen.queryByText("Guardrail Details")).not.toBeInTheDocument();
  });

  it("writes ?garden_tab= when a tab is clicked and removes it when going back to the overview", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderDetail(cardWithEval, undefined, onUrlUpdate);

    await user.click(screen.getByText("Eval Results"));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("garden_tab")).toBe("eval"));
    expect(screen.getByText("F1 Score")).toBeInTheDocument();

    await user.click(screen.getByText("Overview"));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("garden_tab")).toBe(false));
    expect(screen.getByText("Guardrail Details")).toBeInTheDocument();
  });

  it("falls back to the overview and drops ?garden_tab=eval for a card without eval results", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    render(<GuardrailDetailView card={makeCard()} onBack={vi.fn()} accessToken={null} onGuardrailCreated={vi.fn()} />, {
      wrapper: ({ children }) => (
        <NuqsTestingAdapter
          searchParams="?garden_tab=eval&garden_card=bedrock"
          onUrlUpdate={onUrlUpdate}
          hasMemory
          resetUrlUpdateQueueOnMount={false}
        >
          {children}
        </NuqsTestingAdapter>
      ),
    });

    expect(screen.getByText("Guardrail Details")).toBeInTheDocument();
    expect(screen.queryByText("Eval Results")).not.toBeInTheDocument();
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("garden_tab")).toBe(false);
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("garden_card")).toBe("bedrock");
  });
});
