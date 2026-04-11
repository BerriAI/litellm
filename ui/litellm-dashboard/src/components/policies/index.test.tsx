import React from "react";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "../networking";
import PoliciesPanel from "./index";
import type { PolicyAttachment } from "./types";

vi.mock("./impact_popover", () => ({
  default: () => <button type="button" aria-label="View blast radius" />,
}));

vi.mock("@heroicons/react/outline", () => ({
  TrashIcon: function TrashIcon() {
    return null;
  },
  SwitchVerticalIcon: function SwitchVerticalIcon() {
    return null;
  },
  ChevronUpIcon: function ChevronUpIcon() {
    return null;
  },
  ChevronDownIcon: function ChevronDownIcon() {
    return null;
  },
  EyeIcon: function EyeIcon() {
    return null;
  },
}));

vi.mock("@tremor/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tremor/react")>();
  return {
    ...actual,
    Button: React.forwardRef<HTMLButtonElement, React.ComponentProps<"button">>(
      ({ children, ...props }, ref) =>
        React.createElement("button", { ...props, ref, type: props.type ?? "button" }, children)
    ),
    Tooltip: ({ children }: { children?: React.ReactNode }) =>
      React.createElement(React.Fragment, null, children),
    Switch: ({
      checked,
      onChange,
      className,
    }: {
      checked?: boolean;
      onChange?: (v: boolean) => void;
      className?: string;
    }) =>
      React.createElement("input", {
        type: "checkbox",
        role: "switch",
        checked,
        onChange: (e: React.ChangeEvent<HTMLInputElement>) => onChange?.(e.target.checked),
        className,
      }),
    Icon: ({ icon: IconComp, onClick, className }: { icon?: unknown; onClick?: () => void; className?: string }) =>
      React.createElement(
        "span",
        { onClick, className, role: onClick ? "button" : undefined },
        (IconComp as { displayName?: string; name?: string } | undefined)?.displayName ??
          (IconComp as { name?: string } | undefined)?.name ??
          "icon"
      ),
  };
});

vi.mock("./policy_templates", () => ({
  default: () => null,
}));
vi.mock("./policy_table", () => ({
  default: () => null,
}));
vi.mock("./policy_info", () => ({
  default: () => null,
}));
vi.mock("./add_policy_form", () => ({
  default: () => null,
}));
vi.mock("./pipeline_flow_builder", () => ({
  FlowBuilderPage: () => null,
}));
vi.mock("./add_attachment_form", () => ({
  default: () => null,
}));
vi.mock("./policy_test_panel", () => ({
  default: () => null,
}));
vi.mock("./guardrail_selection_modal", () => ({
  default: () => null,
}));
vi.mock("./template_parameter_modal", () => ({
  default: () => null,
}));
vi.mock("./ai_suggestion_modal", () => ({
  default: () => null,
}));

vi.mock("../networking", () => ({
  getPoliciesList: vi.fn(() => Promise.resolve({ policies: [] })),
  deletePolicyCall: vi.fn(),
  getPolicyAttachmentsList: vi.fn(() => Promise.resolve({ attachments: [] as PolicyAttachment[] })),
  deletePolicyAttachmentCall: vi.fn(() => Promise.resolve({})),
  getGuardrailsList: vi.fn(() => Promise.resolve({ guardrails: [] })),
  getPolicyInfo: vi.fn(),
  createPolicyCall: vi.fn(),
  updatePolicyCall: vi.fn(),
  createPolicyAttachmentCall: vi.fn(),
  createGuardrailCall: vi.fn(),
  enrichPolicyTemplate: vi.fn(),
}));

describe("PoliciesPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.getPoliciesList).mockResolvedValue({ policies: [] });
    vi.mocked(networking.getPolicyAttachmentsList).mockResolvedValue({ attachments: [] });
    vi.mocked(networking.getGuardrailsList).mockResolvedValue({ guardrails: [] });
    vi.mocked(networking.deletePolicyAttachmentCall).mockResolvedValue({});
  });

  it("should open delete confirmation and call deletePolicyAttachmentCall when delete is confirmed", async () => {
    const user = userEvent.setup();
    const attachment: PolicyAttachment = {
      attachment_id: "att-delete-int-1",
      policy_name: "policy-under-test",
      scope: null,
      teams: [],
      keys: [],
      models: [],
      tags: [],
    };
    vi.mocked(networking.getPolicyAttachmentsList).mockResolvedValue({ attachments: [attachment] });

    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />);

    await user.click(screen.getByRole("tab", { name: /^Attachments$/i }));

    await screen.findByText("policy-under-test");

    await user.click(screen.getByRole("button", { name: /delete attachment/i }));

    const modal = await screen.findByRole("dialog");
    expect(within(modal).getByText("Delete Attachment")).toBeInTheDocument();
    expect(within(modal).getByText("att-delete-int-1")).toBeInTheDocument();

    await user.click(within(modal).getByRole("button", { name: /^Delete$/i }));

    await waitFor(() => {
      expect(networking.deletePolicyAttachmentCall).toHaveBeenCalledWith("test-token", "att-delete-int-1");
    });
    expect(networking.deletePolicyAttachmentCall).toHaveBeenCalledTimes(1);
  });
});
