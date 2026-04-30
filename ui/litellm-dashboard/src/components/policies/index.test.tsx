import React from "react";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import PoliciesPanel from "./index";

/**
 * Ant Design's static Modal.confirm often does not run onOk in the real app (React 18+).
 * In jsdom it may still run; we mock confirm as a no-op so the test fails until the panel
 * uses a controlled DeleteResourceModal instead of Modal.confirm.
 */
vi.mock("antd", async (importOriginal) => {
  const mod = await importOriginal<typeof import("antd")>();
  return {
    ...mod,
    Modal: Object.assign(mod.Modal, {
      confirm: vi.fn(),
    }),
  };
});

const EXPECTED_ATTACHMENT_ID = "att-11111111-2222-3333-4444-555555555555" as const;

const networkingMocks = vi.hoisted(() => ({
  deletePolicyAttachmentCall: vi.fn().mockResolvedValue(undefined),
  getPoliciesList: vi.fn().mockResolvedValue({ policies: [] }),
  getPolicyAttachmentsList: vi.fn().mockResolvedValue({
    attachments: [
      {
        attachment_id: "att-11111111-2222-3333-4444-555555555555",
        policy_name: "test-policy",
        scope: null,
        teams: [],
        keys: [],
        models: [],
        tags: [],
      },
    ],
  }),
  getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [] }),
  getPolicyInfo: vi.fn().mockResolvedValue({}),
  deletePolicyCall: vi.fn().mockResolvedValue(undefined),
  createPolicyCall: vi.fn(),
  updatePolicyCall: vi.fn(),
  createPolicyAttachmentCall: vi.fn(),
  createGuardrailCall: vi.fn(),
  enrichPolicyTemplate: vi.fn(),
}));

vi.mock("../networking", () => ({
  ...networkingMocks,
}));

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
}));

vi.mock("@tremor/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tremor/react")>();
  return {
    ...actual,
    Button: React.forwardRef<HTMLButtonElement, any>(({ children, ...props }, ref) =>
      React.createElement("button", { ...props, ref }, children),
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
    Icon: ({ icon: _IconComp, onClick, className }: any) =>
      React.createElement(
        "button",
        { type: "button", onClick, className },
        "TrashIcon",
      ),
  };
});

vi.mock("./policy_templates", () => ({
  __esModule: true,
  default: () => <div data-testid="policy-templates-stub" />,
}));

vi.mock("./pipeline_flow_builder", () => ({
  FlowBuilderPage: () => null,
}));

vi.mock("./policy_info", () => ({
  __esModule: true,
  default: () => null,
}));

vi.mock("./add_policy_form", () => ({
  __esModule: true,
  default: () => null,
}));

vi.mock("./guardrail_selection_modal", () => ({
  __esModule: true,
  default: () => null,
}));

vi.mock("./template_parameter_modal", () => ({
  __esModule: true,
  default: () => null,
}));

vi.mock("./ai_suggestion_modal", () => ({
  __esModule: true,
  default: () => null,
}));

vi.mock("./policy_test_panel", () => ({
  __esModule: true,
  default: () => null,
}));

vi.mock("./add_attachment_form", () => ({
  __esModule: true,
  default: () => null,
}));

describe("PoliciesPanel attachment delete", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should call deletePolicyAttachmentCall after the user confirms delete in the attachment modal", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />);

    await waitFor(() => {
      expect(networkingMocks.getPolicyAttachmentsList).toHaveBeenCalled();
    });

    await user.click(screen.getByRole("tab", { name: /^attachments$/i }));

    await waitFor(() => {
      expect(screen.getByText("test-policy")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /TrashIcon/i }));

    const dialog = await screen.findByRole("dialog", {}, { timeout: 5000 });
    expect(
      within(dialog).getByText(/Are you sure you want to delete this attachment/i),
    ).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /^delete$/i }));

    await waitFor(() => {
      expect(networkingMocks.deletePolicyAttachmentCall).toHaveBeenCalledTimes(1);
      expect(networkingMocks.deletePolicyAttachmentCall).toHaveBeenCalledWith("test-token", EXPECTED_ATTACHMENT_ID);
    });
  });

  it("should show mutation pending state while attachment delete is in flight", async () => {
    let resolveDelete: (() => void) | undefined;
    const deletePromise = new Promise<void>((resolve) => {
      resolveDelete = resolve;
    });
    networkingMocks.deletePolicyAttachmentCall.mockImplementationOnce(() => deletePromise);

    const user = userEvent.setup();
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />);

    await waitFor(() => {
      expect(networkingMocks.getPolicyAttachmentsList).toHaveBeenCalled();
    });

    await user.click(screen.getByRole("tab", { name: /^attachments$/i }));
    await waitFor(() => {
      expect(screen.getByText("test-policy")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /TrashIcon/i }));
    const dialog = await screen.findByRole("dialog", {}, { timeout: 5000 });

    const deleteButton = within(dialog).getByRole("button", { name: /^delete$/i });
    await user.click(deleteButton);

    await waitFor(() => {
      expect(within(dialog).getByRole("button", { name: /deleting/i })).toBeDisabled();
    });

    resolveDelete?.();
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });
  });
});
