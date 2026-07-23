import { fireEvent, screen } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import { testPoliciesAndGuardrails } from "@/components/networking";
import ComplianceUI from "./ComplianceUI";

vi.mock("@/components/networking", () => ({
  getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [] }),
  testPoliciesAndGuardrails: vi.fn(),
}));

vi.mock("@/components/policies/PolicySelector", () => ({
  default: () => null,
  getPolicyOptionEntries: () => [],
}));

vi.mock("@/components/llm_calls/chat_completion", () => ({
  makeOpenAIChatCompletionRequest: vi.fn(),
}));

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn();
});

describe("ComplianceUI", () => {
  it("aborts the in-flight batch run when the component unmounts", () => {
    let capturedSignal: AbortSignal | undefined;
    vi.mocked(testPoliciesAndGuardrails).mockImplementation((_accessToken, _body, signal) => {
      capturedSignal = signal;
      return new Promise(() => {});
    });

    const { unmount } = renderWithProviders(<ComplianceUI accessToken="test-token" />);

    fireEvent.click(screen.getAllByText("All")[0]);
    fireEvent.click(screen.getByRole("button", { name: /Simulate/ }));

    expect(testPoliciesAndGuardrails).toHaveBeenCalledTimes(1);
    expect(capturedSignal).toBeDefined();
    expect(capturedSignal?.aborted).toBe(false);

    unmount();

    expect(capturedSignal?.aborted).toBe(true);
  });
});
