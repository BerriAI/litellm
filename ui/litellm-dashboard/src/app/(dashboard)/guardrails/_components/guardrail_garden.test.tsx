import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, renderWithProviders, screen, waitFor } from "@/../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
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

const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0];

describe("GuardrailGarden", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const renderGarden = (searchParams?: string, onUrlUpdate?: OnUrlUpdateFunction) =>
    renderWithProviders(<GuardrailGarden accessToken="test-token" onGuardrailCreated={vi.fn()} />, {
      searchParams,
      onUrlUpdate,
    });

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
    renderGarden();

    const target = PARTNER_CARDS[0];
    fireEvent.change(screen.getByPlaceholderText("Search guardrails"), { target: { value: target.name } });

    expect(await screen.findByText(target.name)).toBeInTheDocument();
    const otherPartner = PARTNER_CARDS.find((c) => c.name !== target.name);
    if (otherPartner) {
      expect(screen.queryByText(otherPartner.name)).not.toBeInTheDocument();
    }
  });

  it("should show an empty result set for a query that matches nothing", async () => {
    renderGarden();

    fireEvent.change(screen.getByPlaceholderText("Search guardrails"), {
      target: { value: "zzzzznotaguardrailzzzzz" },
    });

    expect(await screen.findByText("Show all (0)")).toBeInTheDocument();
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

  describe("URL state", () => {
    it("should filter cards and prefill the search box from ?garden_q=", () => {
      const [target, other] = PARTNER_CARDS;
      renderGarden(`?garden_q=${encodeURIComponent(target.name)}`);

      expect(screen.getByPlaceholderText("Search guardrails")).toHaveValue(target.name);
      expect(screen.getByText(target.name)).toBeInTheDocument();
      expect(screen.queryByText(other.name)).not.toBeInTheDocument();
    });

    it("should write the search query to ?garden_q= and drop it once the box is cleared", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderGarden(undefined, onUrlUpdate);
      const input = screen.getByPlaceholderText("Search guardrails");

      fireEvent.change(input, { target: { value: "bedrock" } });
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("garden_q")).toBe("bedrock"));

      fireEvent.change(input, { target: { value: "" } });
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("garden_q")).toBe(false));
    });

    it("should show every litellm card when ?garden_all=true", () => {
      renderGarden("?garden_all=true");

      expect(screen.getByText("Show less")).toBeInTheDocument();
      expect(screen.getByText(LITELLM_CARDS[LITELLM_CARDS.length - 1].name)).toBeInTheDocument();
    });

    it("should write ?garden_all=true when show all is clicked and remove it on show less", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderGarden(undefined, onUrlUpdate);

      await user.click(screen.getByText(`Show all (${LITELLM_CARDS.length})`));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("garden_all")).toBe("true"));

      await user.click(screen.getByText("Show less"));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("garden_all")).toBe(false));
    });

    it("should open the card named by ?garden_card=", () => {
      const target = PARTNER_CARDS[1];
      renderGarden(`?garden_card=${target.id}`);

      expect(screen.getByText(`Detail for ${target.name}`)).toBeInTheDocument();
      expect(screen.queryByPlaceholderText("Search guardrails")).not.toBeInTheDocument();
    });

    it("should push ?garden_card= and reset the detail tab when a card is clicked", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      const target = PARTNER_CARDS[0];
      renderGarden("?garden_tab=eval", onUrlUpdate);

      await user.click(screen.getByText(target.name));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("garden_card")).toBe(target.id));
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("garden_tab")).toBe(false);
      expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("push");
    });

    it("should clear the card and its detail tab with a history replace when going back", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      const target = LITELLM_CARDS[0];
      renderGarden(`?garden_card=${target.id}&garden_tab=eval&garden_q=x`, onUrlUpdate);

      await user.click(screen.getByRole("button", { name: "Back to garden" }));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("garden_card")).toBe(false));
      const update = lastUrlUpdate(onUrlUpdate);
      expect(update?.searchParams.has("garden_tab")).toBe(false);
      expect(update?.searchParams.get("garden_q")).toBe("x");
      expect(update?.options.history).toBe("replace");
      expect(await screen.findByPlaceholderText("Search guardrails")).toBeInTheDocument();
    });

    it("should show the garden and drop an unknown ?garden_card= from the URL", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      render(<GuardrailGarden accessToken="test-token" onGuardrailCreated={vi.fn()} />, {
        wrapper: ({ children }) => (
          <NuqsTestingAdapter
            searchParams="?garden_card=not-a-card&garden_all=true"
            onUrlUpdate={onUrlUpdate}
            hasMemory
            resetUrlUpdateQueueOnMount={false}
          >
            {children}
          </NuqsTestingAdapter>
        ),
      });

      expect(screen.getByPlaceholderText("Search guardrails")).toBeInTheDocument();
      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      const update = lastUrlUpdate(onUrlUpdate);
      expect(update?.searchParams.has("garden_card")).toBe(false);
      expect(update?.searchParams.get("garden_all")).toBe("true");
      expect(update?.options.history).toBe("replace");
    });
  });
});
