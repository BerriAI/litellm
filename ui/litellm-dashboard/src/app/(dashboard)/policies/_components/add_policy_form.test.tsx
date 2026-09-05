import { cleanup, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../../tests/test-utils";
import type { Policy } from "@/components/policies/types";
import AddPolicyForm from "./add_policy_form";

vi.mock("@/components/networking", () => ({
  getResolvedGuardrails: vi.fn().mockResolvedValue({ resolved_guardrails: [] }),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [{ id: "gpt-4" }] }),
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: { success: vi.fn(), fromBackend: vi.fn() },
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn().mockReturnValue({ userId: "u1", userRole: "Admin" }),
}));

const EXISTING_POLICY: Policy = {
  policy_id: "pol-1",
  policy_name: "existing-policy",
  inherit: "parent-policy",
  description: "an existing policy",
  guardrails_add: ["guard-a"],
  guardrails_remove: ["guard-b"],
  condition: { model: "gpt-4" },
};

const PARENT_POLICY: Policy = {
  policy_id: "pol-parent",
  policy_name: "parent-policy",
  inherit: null,
  description: null,
  guardrails_add: ["guard-c"],
  guardrails_remove: [],
  condition: null,
};

describe("AddPolicyForm", () => {
  const createPolicy = vi.fn().mockResolvedValue({});
  const updatePolicy = vi.fn().mockResolvedValue({});

  const defaultProps = {
    visible: true,
    onClose: vi.fn(),
    onSuccess: vi.fn(),
    onOpenFlowBuilder: vi.fn(),
    accessToken: "test-token",
    existingPolicies: [PARENT_POLICY, EXISTING_POLICY],
    availableGuardrails: [
      { guardrail_id: "g-a", guardrail_name: "guard-a" },
      { guardrail_id: "g-b", guardrail_name: "guard-b" },
      { guardrail_id: "g-c", guardrail_name: "guard-c" },
    ] as never,
    createPolicy,
    updatePolicy,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    cleanup();
  });

  const enterSimpleForm = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(await screen.findByRole("button", { name: "Create Policy" }));
  };

  it("should send exactly six keys with empty-to-undefined and empty-to-array defaults on create", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AddPolicyForm {...defaultProps} />);
    await enterSimpleForm(user);

    fireEvent.change(await screen.findByLabelText("Policy Name"), { target: { value: "brand-new-policy" } });
    await user.click(screen.getByRole("button", { name: "Create Policy" }));

    await waitFor(() => {
      expect(createPolicy).toHaveBeenCalled();
    });
    const payload = createPolicy.mock.calls[0][1];
    expect(Object.keys(payload).sort()).toEqual([
      "condition",
      "description",
      "guardrails_add",
      "guardrails_remove",
      "inherit",
      "policy_name",
    ]);
    expect(payload).toStrictEqual({
      policy_name: "brand-new-policy",
      description: undefined,
      inherit: undefined,
      guardrails_add: [],
      guardrails_remove: [],
      condition: undefined,
    });
  });

  it("should collapse a blank description to undefined rather than an empty string", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AddPolicyForm {...defaultProps} />);
    await enterSimpleForm(user);

    const description = await screen.findByLabelText("Description");
    fireEvent.change(description, { target: { value: "x" } });
    await user.clear(description);
    fireEvent.change(await screen.findByLabelText("Policy Name"), { target: { value: "blank-description" } });
    await user.click(screen.getByRole("button", { name: "Create Policy" }));

    await waitFor(() => {
      expect(createPolicy).toHaveBeenCalled();
    });
    expect(createPolicy.mock.calls[0][1].description).toBeUndefined();
  });

  it("should send the seeded policy through updatePolicy with condition wrapped in a model object", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AddPolicyForm {...defaultProps} editingPolicy={EXISTING_POLICY} />);

    await user.click(await screen.findByRole("button", { name: "Update Policy" }));

    await waitFor(() => {
      expect(updatePolicy).toHaveBeenCalled();
    });
    expect(updatePolicy.mock.calls[0][0]).toBe("test-token");
    expect(updatePolicy.mock.calls[0][1]).toBe("pol-1");
    expect(updatePolicy.mock.calls[0][2]).toStrictEqual({
      policy_name: "existing-policy",
      description: "an existing policy",
      inherit: "parent-policy",
      guardrails_add: ["guard-a"],
      guardrails_remove: ["guard-b"],
      condition: { model: "gpt-4" },
    });
    expect(createPolicy).not.toHaveBeenCalled();
  });

  it("should keep the policy name field disabled while editing", async () => {
    renderWithProviders(<AddPolicyForm {...defaultProps} editingPolicy={EXISTING_POLICY} />);

    expect(await screen.findByLabelText("Policy Name")).toBeDisabled();
  });

  it("should block submission and call neither api when the policy name is missing", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AddPolicyForm {...defaultProps} />);
    await enterSimpleForm(user);

    await user.click(await screen.findByRole("button", { name: "Create Policy" }));

    expect(await screen.findByText("Please enter a policy name")).toBeInTheDocument();
    expect(createPolicy).not.toHaveBeenCalled();
    expect(updatePolicy).not.toHaveBeenCalled();
  });

  it("should block submission when the policy name has characters outside the allowed set", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AddPolicyForm {...defaultProps} />);
    await enterSimpleForm(user);

    fireEvent.change(await screen.findByLabelText("Policy Name"), { target: { value: "not a valid name!" } });
    await user.click(screen.getByRole("button", { name: "Create Policy" }));

    expect(
      await screen.findByText("Policy name can only contain letters, numbers, hyphens, and underscores"),
    ).toBeInTheDocument();
    expect(createPolicy).not.toHaveBeenCalled();
  });

  it("should swap the model condition label and clear the value when the condition type changes", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AddPolicyForm {...defaultProps} editingPolicy={EXISTING_POLICY} />);

    expect(await screen.findByLabelText("Model (Optional)")).toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: "Custom Regex Pattern" }));

    const regexField = await screen.findByLabelText("Regex Pattern (Optional)");
    expect(regexField).toHaveValue("");
    expect(screen.queryByLabelText("Model (Optional)")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Update Policy" }));

    await waitFor(() => {
      expect(updatePolicy).toHaveBeenCalled();
    });
    expect(updatePolicy.mock.calls[0][2].condition).toBeUndefined();
  });

  it("should open the flow builder instead of the simple form when that mode is confirmed", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const onOpenFlowBuilder = vi.fn();
    renderWithProviders(<AddPolicyForm {...defaultProps} onClose={onClose} onOpenFlowBuilder={onOpenFlowBuilder} />);

    await user.click(await screen.findByText("Flow Builder"));

    expect(
      screen.getByText("You'll be taken to the Flow Builder to design your policy logic visually."),
    ).toBeInTheDocument();
    expect(screen.queryByText(/full-screen/i)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Continue to Builder" }));

    expect(onOpenFlowBuilder).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(createPolicy).not.toHaveBeenCalled();
  });
});
