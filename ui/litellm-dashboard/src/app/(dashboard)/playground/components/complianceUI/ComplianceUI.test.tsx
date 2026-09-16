import userEvent from "@testing-library/user-event";
import { QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { PropsWithChildren } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, renderWithProviders, screen, testQueryClient, waitFor } from "@/../tests/test-utils";
import ComplianceUI from "./ComplianceUI";

const getGuardrailsList = vi.fn();
const testPoliciesAndGuardrails = vi.fn();
const canViewPolicies = vi.fn(() => true);

vi.mock("@/components/networking", () => ({
  getGuardrailsList: (...args: unknown[]) => getGuardrailsList(...args),
  testPoliciesAndGuardrails: (...args: unknown[]) => testPoliciesAndGuardrails(...args),
}));

vi.mock("@/app/(dashboard)/hooks/useCan", () => ({
  default: () => canViewPolicies(),
}));

vi.mock("@/components/policies/PolicySelector", () => ({
  default: ({ value, onChange }: { value: string[]; onChange: (policies: string[]) => void }) => (
    <div>
      <span>{`Chosen policies: ${value.join("|")}`}</span>
      <button type="button" onClick={() => onChange(["pii-policy", "toxicity-policy"])}>
        Choose policies
      </button>
    </div>
  ),
  getPolicyOptionEntries: () => [],
}));

vi.mock("@/components/llm_calls/chat_completion", () => ({
  makeOpenAIChatCompletionRequest: vi.fn().mockResolvedValue(undefined),
}));

type RenderOptions = { searchParams?: string; onUrlUpdate?: OnUrlUpdateFunction; persistInUrl?: boolean };

const renderCompliance = ({ searchParams, onUrlUpdate, persistInUrl }: RenderOptions = {}) =>
  renderWithProviders(<ComplianceUI accessToken="sk-test" persistInUrl={persistInUrl} />, {
    searchParams,
    onUrlUpdate,
  });

const renderComplianceKeepingMountUpdates = (searchParams: string, onUrlUpdate: OnUrlUpdateFunction) => {
  const Providers = ({ children }: PropsWithChildren) => (
    <NuqsTestingAdapter
      searchParams={searchParams}
      onUrlUpdate={onUrlUpdate}
      hasMemory
      resetUrlUpdateQueueOnMount={false}
    >
      <QueryClientProvider client={testQueryClient}>{children}</QueryClientProvider>
    </NuqsTestingAdapter>
  );
  return render(<ComplianceUI accessToken="sk-test" />, { wrapper: Providers });
};

const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) => {
  const update = onUrlUpdate.mock.calls.at(-1)?.[0];
  if (!update) throw new Error("expected a URL update");
  return update;
};

const QUICK_TEST_HINT = "Type a prompt below to quickly test it.";

beforeEach(() => {
  vi.clearAllMocks();
  canViewPolicies.mockReturnValue(true);
  Element.prototype.scrollIntoView = vi.fn();
  getGuardrailsList.mockResolvedValue({
    guardrails: [{ guardrail_name: "pii-mask" }, { guardrail_name: "toxicity-filter" }],
  });
  testPoliciesAndGuardrails.mockResolvedValue({ inputs: { texts: ["hello"] }, guardrail_errors: [] });
});

describe("ComplianceUI URL state", () => {
  it("shows the quick test panel by default", () => {
    renderCompliance();

    expect(screen.getByText(QUICK_TEST_HINT)).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Results" })).not.toBeInTheDocument();
  });

  it("opens the batch results panel named in the URL", () => {
    renderCompliance({ searchParams: "?cmpl_tab=batch" });

    expect(screen.getByRole("heading", { name: "Results" })).toBeInTheDocument();
    expect(screen.queryByText(QUICK_TEST_HINT)).not.toBeInTheDocument();
  });

  it("falls back to the quick test panel for an unknown URL tab and clears it", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderComplianceKeepingMountUpdates("?cmpl_tab=history", onUrlUpdate);

    expect(screen.getByText(QUICK_TEST_HINT)).toBeInTheDocument();
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.has("cmpl_tab")).toBe(false));
  });

  it("writes the chosen panel to the URL", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderCompliance({ onUrlUpdate });

    await user.click(screen.getByRole("button", { name: /Batch Results/ }));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.get("cmpl_tab")).toBe("batch"));
    expect(screen.getByRole("heading", { name: "Results" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Quick Test/ }));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.has("cmpl_tab")).toBe(false));
  });

  it("restores policies and guardrails from the URL", async () => {
    renderCompliance({ searchParams: "?cmpl_policies=pii-policy&cmpl_guardrails=pii-mask,custom-guard" });

    expect(screen.getByText("Chosen policies: pii-policy")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /2 selected/ })).toBeInTheDocument();
    expect(screen.getByText("Testing against:")).toBeInTheDocument();
    expect(screen.getAllByText("custom-guard")).toHaveLength(2);
    expect(await screen.findAllByText("pii-mask")).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Test 1 policy & 2 guardrails" })).toBeDisabled();
  });

  it("tests a prompt against the policies and guardrails from the URL", async () => {
    const user = userEvent.setup();
    renderCompliance({ searchParams: "?cmpl_policies=pii-policy&cmpl_guardrails=pii-mask" });

    fireEvent.change(screen.getByPlaceholderText("Enter text to test..."), { target: { value: "hello" } });
    await user.click(screen.getByRole("button", { name: "Test 1 policy & 1 guardrail" }));

    await waitFor(() => expect(testPoliciesAndGuardrails).toHaveBeenCalledTimes(1));
    expect(testPoliciesAndGuardrails.mock.calls[0][1]).toMatchObject({
      policy_names: ["pii-policy"],
      guardrail_names: ["pii-mask"],
    });
  });

  it("ignores URL policies for a user who cannot view policies", async () => {
    const user = userEvent.setup();
    canViewPolicies.mockReturnValue(false);
    renderCompliance({ searchParams: "?cmpl_policies=pii-policy&cmpl_guardrails=pii-mask" });

    fireEvent.change(screen.getByPlaceholderText("Enter text to test..."), { target: { value: "hello" } });
    await user.click(screen.getByRole("button", { name: "Test 1 guardrail" }));

    await waitFor(() => expect(testPoliciesAndGuardrails).toHaveBeenCalledTimes(1));
    expect(testPoliciesAndGuardrails.mock.calls[0][1].policy_names).toBeUndefined();
    expect(testPoliciesAndGuardrails.mock.calls[0][1].guardrail_names).toEqual(["pii-mask"]);
    expect(screen.queryByText("pii-policy")).not.toBeInTheDocument();
  });

  it("writes chosen policies and guardrails to the URL and clears them on reset", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderCompliance({ onUrlUpdate });

    await user.click(screen.getByRole("button", { name: "Choose policies" }));
    await waitFor(() =>
      expect(lastUrlUpdate(onUrlUpdate).searchParams.get("cmpl_policies")).toBe("pii-policy,toxicity-policy"),
    );

    await user.click(screen.getByRole("button", { name: /None selected/ }));
    await user.click(await screen.findByRole("button", { name: /toxicity-filter/ }));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.get("cmpl_guardrails")).toBe("toxicity-filter"));

    await user.click(screen.getByRole("button", { name: /^pii-mask/ }));
    await waitFor(() =>
      expect(lastUrlUpdate(onUrlUpdate).searchParams.get("cmpl_guardrails")).toBe("toxicity-filter,pii-mask"),
    );

    await user.click(screen.getByRole("button", { name: /^toxicity-filter/ }));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.get("cmpl_guardrails")).toBe("pii-mask"));

    await user.click(screen.getByRole("button", { name: "Reset" }));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.has("cmpl_guardrails")).toBe(false));
    expect(lastUrlUpdate(onUrlUpdate).searchParams.has("cmpl_policies")).toBe(false);
    expect(screen.getByText("Chosen policies:")).toBeInTheDocument();
  });

  it("keeps its state out of the URL when embedded", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderCompliance({
      searchParams: "?cmpl_tab=batch&cmpl_policies=pii-policy&cmpl_guardrails=pii-mask",
      onUrlUpdate,
      persistInUrl: false,
    });

    expect(screen.getByText(QUICK_TEST_HINT)).toBeInTheDocument();
    expect(screen.getByText("Chosen policies:")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /None selected/ })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Choose policies" }));
    await user.click(screen.getByRole("button", { name: /None selected/ }));
    await user.click(await screen.findByRole("button", { name: /toxicity-filter/ }));
    await user.click(screen.getByRole("button", { name: /Batch Results/ }));

    expect(screen.getByText("Chosen policies: pii-policy|toxicity-policy")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /1 selected/ })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Results" })).toBeInTheDocument();
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });
});
