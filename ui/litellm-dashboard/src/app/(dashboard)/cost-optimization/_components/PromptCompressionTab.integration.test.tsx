import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import PromptCompressionTab from "./PromptCompressionTab";

const createGuardrailCall = vi.fn();
const getGuardrailsList = vi.fn();

vi.mock("@/components/networking", () => ({
  createGuardrailCall: (...args: unknown[]) => createGuardrailCall(...args),
  getGuardrailsList: (...args: unknown[]) => getGuardrailsList(...args),
}));

const submittedPayload = (): Record<string, unknown> => {
  expect(createGuardrailCall).toHaveBeenCalledTimes(1);
  return createGuardrailCall.mock.calls[0][1] as Record<string, unknown>;
};

describe("PromptCompressionTab submit payload", () => {
  beforeEach(() => {
    createGuardrailCall.mockClear().mockResolvedValue({});
    getGuardrailsList.mockClear().mockResolvedValue({ guardrails: [] });
  });

  it("sends the trimmed name and api base with default_on true", async () => {
    const user = userEvent.setup();
    render(<PromptCompressionTab accessToken="test-token" />);

    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "  headroom-compression  " } });
    fireEvent.change(screen.getByLabelText("Headroom API base"), {
      target: { value: "  https://headroom.example.com  " },
    });
    await user.click(screen.getByRole("button", { name: "Add guardrail" }));

    await vi.waitFor(() =>
      expect(submittedPayload()).toEqual({
        guardrail_name: "headroom-compression",
        litellm_params: {
          guardrail: "headroom",
          mode: "pre_call",
          api_base: "https://headroom.example.com",
          default_on: true,
        },
      }),
    );
    expect(createGuardrailCall.mock.calls[0][0]).toBe("test-token");
  });

  it("sends default_on false once the apply-to-all switch is turned off", async () => {
    const user = userEvent.setup();
    render(<PromptCompressionTab accessToken="test-token" />);

    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "headroom-optin" } });
    fireEvent.change(screen.getByLabelText("Headroom API base"), { target: { value: "https://headroom.example.com" } });
    await user.click(screen.getByLabelText("Apply to all requests"));
    await user.click(screen.getByRole("button", { name: "Add guardrail" }));

    await vi.waitFor(() =>
      expect(submittedPayload()).toEqual({
        guardrail_name: "headroom-optin",
        litellm_params: {
          guardrail: "headroom",
          mode: "pre_call",
          api_base: "https://headroom.example.com",
          default_on: false,
        },
      }),
    );
  });

  it("blocks submission and shows both required messages when the form is empty", async () => {
    const user = userEvent.setup();
    render(<PromptCompressionTab accessToken="test-token" />);

    await user.click(screen.getByRole("button", { name: "Add guardrail" }));

    expect(await screen.findByText("Name is required")).toBeInTheDocument();
    expect(screen.getByText("API base is required")).toBeInTheDocument();
    expect(createGuardrailCall).not.toHaveBeenCalled();
  });

  it("submits when Enter is pressed inside a text field", async () => {
    const user = userEvent.setup();
    render(<PromptCompressionTab accessToken="test-token" />);

    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "headroom-compression" } });
    await user.type(screen.getByLabelText("Headroom API base"), "https://headroom.example.com{Enter}");

    await vi.waitFor(() => expect(createGuardrailCall).toHaveBeenCalledTimes(1));
  });

  it("clears the name and restores the default switch state after a successful create", async () => {
    const user = userEvent.setup();
    render(<PromptCompressionTab accessToken="test-token" />);

    fireEvent.change(screen.getByLabelText("Name"), { target: { value: "headroom-compression" } });
    fireEvent.change(screen.getByLabelText("Headroom API base"), { target: { value: "https://headroom.example.com" } });
    await user.click(screen.getByLabelText("Apply to all requests"));
    await user.click(screen.getByRole("button", { name: "Add guardrail" }));

    await vi.waitFor(() => expect(screen.getByLabelText("Name")).toHaveValue(""));
    expect(screen.getByLabelText("Headroom API base")).toHaveValue("");
    expect(screen.getByLabelText("Apply to all requests")).toBeChecked();
    expect(getGuardrailsList).toHaveBeenCalledTimes(2);
  });

  it("keeps the always-on and opt-in badges for the guardrails it lists", async () => {
    getGuardrailsList.mockResolvedValue({
      guardrails: [
        {
          guardrail_id: "g-1",
          guardrail_name: "always-on-one",
          litellm_params: { guardrail: "headroom", api_base: "https://a.example.com", default_on: true },
        },
        {
          guardrail_id: "g-2",
          guardrail_name: "opt-in-one",
          litellm_params: { guardrail: "headroom", api_base: "https://b.example.com", default_on: false },
        },
      ],
    });
    render(<PromptCompressionTab accessToken="test-token" />);

    expect(await screen.findByText("Always on")).toBeInTheDocument();
    expect(screen.getByText("Opt-in")).toBeInTheDocument();
    expect(screen.getByText("https://a.example.com")).toBeInTheDocument();
  });
});
