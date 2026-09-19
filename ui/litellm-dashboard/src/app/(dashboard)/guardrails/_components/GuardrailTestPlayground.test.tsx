import { fireEvent, renderWithProviders, screen, waitFor, within } from "@/../tests/test-utils";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import GuardrailTestPlayground from "./GuardrailTestPlayground";

vi.mock("@/components/networking");

Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

const makeGuardrail = (id: string, name: string) => ({
  guardrail_id: id,
  guardrail_name: name,
  litellm_params: {
    guardrail: "presidio",
    mode: "pre_call" as const,
    default_on: false,
  },
  guardrail_info: {},
});

const sidebarItem = (name: string) => within(screen.getAllByRole("list")[0]).getByText(name);

const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0];

describe("GuardrailTestPlayground", () => {
  const mockAccessToken = "test-token";
  const mockGuardrails = [makeGuardrail("guard-1", "test-guardrail")];
  const twoGuardrails = [makeGuardrail("guard-1", "pii, strict"), makeGuardrail("guard-2", "toxicity")];

  const renderPlayground = (
    guardrailsList = mockGuardrails,
    searchParams?: string,
    onUrlUpdate?: OnUrlUpdateFunction,
  ) =>
    renderWithProviders(
      <GuardrailTestPlayground
        guardrailsList={guardrailsList}
        isLoading={false}
        accessToken={mockAccessToken}
        onClose={vi.fn()}
      />,
      { searchParams, onUrlUpdate },
    );

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should allow selecting a guardrail and show test panel", async () => {
    const user = userEvent.setup();

    renderPlayground();

    expect(screen.getByText("Select Guardrails to Test")).toBeInTheDocument();

    await user.click(screen.getByText("test-guardrail"));

    await waitFor(() => {
      expect(screen.getByText("Test Guardrails:")).toBeInTheDocument();
      expect(screen.getByPlaceholderText("Enter text to test with guardrails...")).toBeInTheDocument();
    });

    expect(screen.getByText("1 of 1 selected")).toBeInTheDocument();
  });

  describe("URL state", () => {
    it("should select the guardrails named in ?test_guardrails=, including names with commas", () => {
      renderPlayground(twoGuardrails, `?test_guardrails=${encodeURIComponent("pii%2C strict")},toxicity`);

      expect(screen.getByText("2 of 2 selected")).toBeInTheDocument();
      expect(screen.getByText("Test 2 guardrails")).toBeInTheDocument();
    });

    it("should ignore names in ?test_guardrails= that are not in the guardrail list", () => {
      renderPlayground(twoGuardrails, "?test_guardrails=deleted-guardrail");

      expect(screen.getByText("0 of 2 selected")).toBeInTheDocument();
      expect(screen.getByText("Select Guardrails to Test")).toBeInTheDocument();
    });

    it("should write each selection to ?test_guardrails= and remove the key once nothing is selected", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderPlayground(twoGuardrails, "?test_guardrails=deleted-guardrail", onUrlUpdate);

      await user.click(sidebarItem("pii, strict"));
      await waitFor(() =>
        expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("test_guardrails")).toBe("pii%2C strict"),
      );

      await user.click(sidebarItem("toxicity"));
      await waitFor(() =>
        expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("test_guardrails")).toBe("pii%2C strict,toxicity"),
      );
      expect(screen.getByText("2 of 2 selected")).toBeInTheDocument();

      await user.click(sidebarItem("pii, strict"));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("test_guardrails")).toBe("toxicity"));

      await user.click(sidebarItem("toxicity"));
      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("test_guardrails")).toBe(false));
      expect(screen.getByText("Select Guardrails to Test")).toBeInTheDocument();
    });

    it("should filter the list and prefill the search box from ?test_q=", () => {
      renderPlayground(twoGuardrails, "?test_q=TOX");

      expect(screen.getByPlaceholderText("Search guardrails...")).toHaveValue("TOX");
      expect(screen.getByText("toxicity")).toBeInTheDocument();
      expect(screen.queryByText("pii, strict")).not.toBeInTheDocument();
    });

    it("should write the search box to ?test_q=", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderPlayground(twoGuardrails, undefined, onUrlUpdate);

      fireEvent.change(screen.getByPlaceholderText("Search guardrails..."), { target: { value: "nothing" } });

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("test_q")).toBe("nothing"));
      expect(screen.getByText("No guardrails match your search")).toBeInTheDocument();
    });
  });
});
