import React from "react";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import PoliciesPanel from "./index";
import * as networking from "../networking";
import MessageManager from "@/components/molecules/message_manager";

vi.mock("../networking", () => ({
  getPoliciesList: vi.fn(),
  deletePolicyCall: vi.fn(),
  getPolicyAttachmentsList: vi.fn(),
  deletePolicyAttachmentCall: vi.fn(),
  getGuardrailsList: vi.fn(),
  getPolicyInfo: vi.fn(),
  createPolicyCall: vi.fn(),
  updatePolicyCall: vi.fn(),
  createPolicyAttachmentCall: vi.fn(),
  createGuardrailCall: vi.fn(),
  enrichPolicyTemplate: vi.fn(),
}));

vi.mock("@/components/molecules/message_manager", () => ({
  default: {
    success: vi.fn(),
    error: vi.fn(),
  },
}));

vi.mock("@/utils/roles", () => ({
  isAdminRole: vi.fn(() => true),
}));

vi.mock("@tremor/react", () => ({
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  TabGroup: ({ children }: any) => <div>{children}</div>,
  TabList: ({ children }: any) => <div>{children}</div>,
  Tab: ({ children }: any) => <button>{children}</button>,
  TabPanels: ({ children }: any) => <div>{children}</div>,
  TabPanel: ({ children }: any) => <div>{children}</div>,
}));

vi.mock("antd", () => ({
  Alert: ({ message }: { message: React.ReactNode }) => <div>{message}</div>,
}));

vi.mock("./policy_table", () => ({
  default: () => <div>policy-table</div>,
}));

vi.mock("./policy_info", () => ({
  default: () => <div>policy-info</div>,
}));

vi.mock("./add_policy_form", () => ({
  default: () => null,
}));

vi.mock("./pipeline_flow_builder", () => ({
  FlowBuilderPage: () => <div>flow-builder</div>,
}));

vi.mock("./attachment_table", () => ({
  default: ({
    attachments,
    onDeleteClick,
  }: {
    attachments: Array<{ attachment_id: string }>;
    onDeleteClick: (attachmentId: string) => void;
  }) => (
    <div>
      <div data-testid="attachment-count">{attachments.length}</div>
      <button
        type="button"
        onClick={() => {
          if (attachments.length > 0) {
            onDeleteClick(attachments[0].attachment_id);
          }
        }}
      >
        Trigger attachment delete
      </button>
    </div>
  ),
}));

vi.mock("./add_attachment_form", () => ({
  default: () => null,
}));

vi.mock("./policy_test_panel", () => ({
  default: () => <div>policy-test-panel</div>,
}));

vi.mock("./policy_templates", () => ({
  default: () => <div>policy-templates</div>,
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

vi.mock("../common_components/DeleteResourceModal", () => ({
  default: ({
    isOpen,
    title,
    onOk,
  }: {
    isOpen: boolean;
    title: string;
    onOk: () => void;
  }) =>
    isOpen ? (
      <div>
        <div>{title}</div>
        <button type="button" onClick={onOk}>
          Confirm attachment delete
        </button>
      </div>
    ) : null,
}));

describe("PoliciesPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.getPoliciesList).mockResolvedValue({
      policies: [
        {
          policy_id: "policy-1",
          policy_name: "policy-one",
          inherit: null,
          description: null,
          guardrails_add: [],
          guardrails_remove: [],
          condition: null,
        },
      ],
    } as any);
    vi.mocked(networking.getPolicyAttachmentsList).mockResolvedValue({
      attachments: [
        {
          attachment_id: "att-1",
          policy_name: "policy-one",
          scope: "*",
          teams: [],
          keys: [],
          models: [],
          tags: [],
        },
      ],
    } as any);
    vi.mocked(networking.getGuardrailsList).mockResolvedValue({
      guardrails: [],
    } as any);
    vi.mocked(networking.deletePolicyAttachmentCall).mockResolvedValue({} as any);
  });

  it("should open attachment delete modal and call delete API after confirmation", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />);

    await waitFor(() => {
      expect(screen.getByTestId("attachment-count")).toHaveTextContent("1");
    });

    await user.click(screen.getByRole("button", { name: /trigger attachment delete/i }));
    expect(screen.getByText("Delete Attachment")).toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: /confirm attachment delete/i }),
    );

    await waitFor(() => {
      expect(networking.deletePolicyAttachmentCall).toHaveBeenCalledWith(
        "test-token",
        "att-1",
      );
    });
    expect(MessageManager.success).toHaveBeenCalledWith(
      "Attachment deleted successfully",
    );
  });
});
