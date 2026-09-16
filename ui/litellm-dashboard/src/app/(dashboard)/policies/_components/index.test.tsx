import React from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Policy } from "@/components/policies/types";
import PoliciesPanel from "./index";

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

vi.mock("@/components/networking", () => ({
  ...networkingMocks,
}));

vi.mock("./impact_popover", () => ({
  default: () => <button type="button" aria-label="View blast radius" />,
}));

vi.mock("./policy_templates", () => ({
  __esModule: true,
  default: ({ onUseTemplate }: { onUseTemplate: (template: unknown) => void }) => (
    <button
      type="button"
      onClick={() =>
        onUseTemplate({
          id: "tpl-pii",
          guardrailDefinitions: [],
          templateData: { policy_name: "template-pii-policy", guardrails_add: [], guardrails_remove: [] },
        })
      }
    >
      Use stub template
    </button>
  ),
}));

vi.mock("./pipeline_flow_builder", () => ({
  FlowBuilderPage: ({
    onBack,
    editingPolicy,
    onVersionCreated,
    onSelectVersion,
    onVersionStatusUpdated,
  }: {
    onBack: () => void;
    editingPolicy?: Policy | null;
    onVersionCreated?: (policy: Policy) => void;
    onSelectVersion?: (policy: Policy) => void;
    onVersionStatusUpdated?: (policy: Policy) => void;
  }) => {
    const draftVersion: Policy = {
      policy_id: "pol-draft-v2",
      policy_name: "pii-policy",
      inherit: null,
      description: null,
      guardrails_add: [],
      guardrails_remove: [],
      condition: null,
      version_number: 2,
      version_status: "draft",
    };
    return (
      <div>
        <div data-testid="flow-builder-target">
          {editingPolicy ? `${editingPolicy.policy_id} ${editingPolicy.version_status ?? "unversioned"}` : "new policy"}
        </div>
        <button type="button" onClick={onBack}>
          Back to policies
        </button>
        <button type="button" onClick={() => onVersionCreated?.(draftVersion)}>
          Create draft version
        </button>
        <button type="button" onClick={() => onSelectVersion?.(draftVersion)}>
          Select draft version
        </button>
        <button
          type="button"
          onClick={() => editingPolicy && onVersionStatusUpdated?.({ ...editingPolicy, version_status: "published" })}
        >
          Publish version
        </button>
      </div>
    );
  },
}));

vi.mock("./policy_info", () => {
  const policyFromInfo = (policyId: string): Policy => ({
    policy_id: policyId,
    policy_name: "pii-policy",
    inherit: null,
    description: null,
    guardrails_add: [],
    guardrails_remove: [],
    condition: null,
  });
  return {
    __esModule: true,
    default: ({
      policyId,
      onClose,
      onEdit,
    }: {
      policyId: string;
      onClose: () => void;
      onEdit: (policy: Policy) => void;
    }) => (
      <div>
        <div data-testid="policy-info">{policyId}</div>
        <button type="button" onClick={onClose}>
          Close policy info
        </button>
        <button type="button" onClick={() => onEdit(policyFromInfo(policyId))}>
          Edit from policy info
        </button>
      </div>
    ),
  };
});

vi.mock("./add_policy_form", () => ({
  __esModule: true,
  default: ({
    visible,
    editingPolicy,
    onOpenFlowBuilder,
  }: {
    visible: boolean;
    editingPolicy?: Policy | null;
    onOpenFlowBuilder: () => void;
  }) =>
    visible ? (
      <div>
        <div data-testid="add-policy-prefill">{editingPolicy?.policy_name ?? "empty"}</div>
        <button type="button" onClick={onOpenFlowBuilder}>
          Open flow builder
        </button>
      </div>
    ) : null,
}));

vi.mock("./guardrail_selection_modal", () => ({
  __esModule: true,
  default: ({ visible, onConfirm }: { visible: boolean; onConfirm: (definitions: unknown[]) => void }) =>
    visible ? (
      <button type="button" onClick={() => onConfirm([])}>
        Confirm guardrails
      </button>
    ) : null,
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

const POLICY_ID = "pol-11111111-2222-3333-4444-555555555555";

const PII_POLICY: Policy = {
  policy_id: POLICY_ID,
  policy_name: "pii-policy",
  inherit: null,
  description: null,
  guardrails_add: [],
  guardrails_remove: [],
  condition: null,
  definition_location: "db",
  version_number: 1,
  version_status: "production",
};

type UrlUpdateMock = ReturnType<typeof vi.fn<OnUrlUpdateFunction>>;

const lastUrlUpdate = (onUrlUpdate: UrlUpdateMock) => onUrlUpdate.mock.calls.at(-1)?.[0];

const renderPanelKeepingMountUpdates = (searchParams: string, onUrlUpdate: UrlUpdateMock) =>
  render(<PoliciesPanel accessToken="test-token" userRole="Admin" />, {
    wrapper: ({ children }: { children: React.ReactNode }) => (
      <NuqsTestingAdapter
        searchParams={searchParams}
        onUrlUpdate={onUrlUpdate}
        hasMemory
        resetUrlUpdateQueueOnMount={false}
      >
        <QueryClientProvider client={testQueryClient}>{children}</QueryClientProvider>
      </NuqsTestingAdapter>
    ),
  });

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

    await user.click(screen.getByTestId(`attachment-actions-${EXPECTED_ATTACHMENT_ID}`));
    await user.click(await screen.findByTestId("attachment-action-delete"));

    const dialog = await screen.findByRole("dialog", {}, { timeout: 5000 });
    expect(within(dialog).getByText(/Are you sure you want to delete this attachment/i)).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /^delete$/i }));

    await waitFor(() => {
      expect(networkingMocks.deletePolicyAttachmentCall).toHaveBeenCalledTimes(1);
    });
    expect(networkingMocks.deletePolicyAttachmentCall).toHaveBeenCalledWith("test-token", EXPECTED_ATTACHMENT_ID);
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

    await user.click(screen.getByTestId(`attachment-actions-${EXPECTED_ATTACHMENT_ID}`));
    await user.click(await screen.findByTestId("attachment-action-delete"));
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

describe("PoliciesPanel ?tab=", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("opens on Templates when the URL has no tab", () => {
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />);

    expect(screen.getByRole("tab", { name: "Templates" })).toHaveAttribute("aria-selected", "true");
  });

  it("activates the tab named in ?tab=", () => {
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />, {
      searchParams: "?tab=attachments",
    });

    expect(screen.getByRole("tab", { name: "Attachments" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Templates" })).toHaveAttribute("aria-selected", "false");
  });

  it("writes the clicked tab to ?tab=", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />, { onUrlUpdate });

    await user.click(screen.getByRole("tab", { name: "Policy Simulator" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("simulator"));
    expect(screen.getByRole("tab", { name: "Policy Simulator" })).toHaveAttribute("aria-selected", "true");
  });

  it("falls back to Templates and clears an unknown ?tab= value", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderPanelKeepingMountUpdates("?tab=bogus", onUrlUpdate);

    expect(screen.getByRole("tab", { name: "Templates" })).toHaveAttribute("aria-selected", "true");
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("tab")).toBe(false);
  });

  it("switches to the Policies tab once a template's guardrails are created", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />, { onUrlUpdate });

    await user.click(screen.getByRole("button", { name: "Use stub template" }));
    await user.click(await screen.findByRole("button", { name: "Confirm guardrails" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("policies"));
    expect(screen.getByRole("tab", { name: "Policies" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByTestId("add-policy-prefill")).toHaveTextContent("template-pii-policy");
  });
});

describe("PoliciesPanel ?policy= detail", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    networkingMocks.getPoliciesList.mockResolvedValue({ policies: [PII_POLICY] });
  });

  afterEach(() => {
    networkingMocks.getPoliciesList.mockResolvedValue({ policies: [] });
  });

  it("opens the policy named in ?policy= instead of the table", () => {
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />, {
      searchParams: `?tab=policies&policy=${POLICY_ID}`,
    });

    expect(screen.getByTestId("policy-info")).toHaveTextContent(POLICY_ID);
    expect(screen.queryByRole("button", { name: "pii-policy" })).not.toBeInTheDocument();
  });

  it("pushes ?policy= when a policy name is clicked", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />, {
      searchParams: "?tab=policies",
      onUrlUpdate,
    });

    await user.click(await screen.findByRole("button", { name: "pii-policy" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("policy")).toBe(POLICY_ID));
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("push");
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("policies");
    expect(screen.getByTestId("policy-info")).toHaveTextContent(POLICY_ID);
  });

  it("removes ?policy= and shows the table again when the detail is closed", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />, {
      searchParams: `?tab=policies&policy=${POLICY_ID}`,
      onUrlUpdate,
    });

    await user.click(screen.getByRole("button", { name: "Close policy info" }));

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("policy")).toBe(false);
    expect(await screen.findByRole("button", { name: "pii-policy" })).toBeInTheDocument();
  });

  it("swaps ?policy= for ?edit_policy= in one update when editing from the detail view", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />, {
      searchParams: `?tab=policies&policy=${POLICY_ID}`,
      onUrlUpdate,
    });

    await user.click(screen.getByRole("button", { name: "Edit from policy info" }));

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalledTimes(1));
    const update = lastUrlUpdate(onUrlUpdate);
    expect(update?.searchParams.get("edit_policy")).toBe(POLICY_ID);
    expect(update?.searchParams.has("policy")).toBe(false);
    expect(update?.options.history).toBe("push");
    expect(await screen.findByTestId("flow-builder-target")).toHaveTextContent(POLICY_ID);
  });
});

describe("PoliciesPanel flow builder", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    networkingMocks.getPoliciesList.mockResolvedValue({ policies: [PII_POLICY] });
  });

  afterEach(() => {
    networkingMocks.getPoliciesList.mockResolvedValue({ policies: [] });
  });

  it("replaces the tabs and policy table with the flow builder while editing, then restores them on back", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />, { onUrlUpdate });

    await user.click(screen.getByRole("tab", { name: /^policies$/i }));
    await user.click(await screen.findByTestId(`policy-actions-${POLICY_ID}`));
    await user.click(await screen.findByTestId("policy-action-edit"));

    expect(await screen.findByRole("button", { name: "Back to policies" })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /^policies$/i })).not.toBeInTheDocument();
    expect(screen.queryByText("pii-policy")).not.toBeInTheDocument();
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("edit_policy")).toBe(POLICY_ID));
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("push");

    await user.click(screen.getByRole("button", { name: "Back to policies" }));

    expect(await screen.findByText("pii-policy")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /^policies$/i })).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByRole("button", { name: "Back to policies" })).not.toBeInTheDocument();
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("edit_policy")).toBe(false));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("policies");
  });

  it("shows a loading state for ?edit_policy= until the policy list arrives, then opens that policy", async () => {
    let resolvePolicies: (value: { policies: Policy[] }) => void = () => {};
    networkingMocks.getPoliciesList.mockReturnValueOnce(
      new Promise((resolve) => {
        resolvePolicies = resolve;
      }),
    );
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />, {
      searchParams: `?tab=policies&edit_policy=${POLICY_ID}`,
    });

    expect(screen.getByRole("status")).toHaveTextContent("Loading policy");
    expect(screen.queryByRole("tab")).not.toBeInTheDocument();
    expect(screen.queryByTestId("flow-builder-target")).not.toBeInTheDocument();

    resolvePolicies({ policies: [PII_POLICY] });

    expect(await screen.findByTestId("flow-builder-target")).toHaveTextContent(`${POLICY_ID} production`);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("drops an ?edit_policy= id that is not in the loaded list and shows the tabs", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderPanelKeepingMountUpdates("?tab=policies&edit_policy=pol-deleted", onUrlUpdate);

    expect(await screen.findByRole("tab", { name: "Policies" })).toHaveAttribute("aria-selected", "true");
    expect(screen.queryByTestId("flow-builder-target")).not.toBeInTheDocument();
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("edit_policy")).toBe(false));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("policies");
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("replace");
  });

  it("keeps a freshly created version open before the refreshed list includes it", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />, {
      searchParams: `?tab=policies&edit_policy=${POLICY_ID}`,
      onUrlUpdate,
    });
    await screen.findByTestId("flow-builder-target");

    networkingMocks.getPoliciesList.mockReturnValue(new Promise(() => {}));
    await user.click(screen.getByRole("button", { name: "Create draft version" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("edit_policy")).toBe("pol-draft-v2"));
    expect(screen.getByTestId("flow-builder-target")).toHaveTextContent("pol-draft-v2 draft");
  });

  it("pushes ?edit_policy= for a version picked in the builder", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />, {
      searchParams: `?tab=policies&edit_policy=${POLICY_ID}`,
      onUrlUpdate,
    });
    await screen.findByTestId("flow-builder-target");

    await user.click(screen.getByRole("button", { name: "Select draft version" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("edit_policy")).toBe("pol-draft-v2"));
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("push");
    expect(screen.getByTestId("flow-builder-target")).toHaveTextContent("pol-draft-v2 draft");
  });

  it("shows the updated version status right away while the list refreshes", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />, {
      searchParams: `?tab=policies&edit_policy=${POLICY_ID}`,
    });
    expect(await screen.findByTestId("flow-builder-target")).toHaveTextContent(`${POLICY_ID} production`);

    networkingMocks.getPoliciesList.mockReturnValue(new Promise(() => {}));
    await user.click(screen.getByRole("button", { name: "Publish version" }));

    expect(screen.getByTestId("flow-builder-target")).toHaveTextContent(`${POLICY_ID} published`);
  });

  it("prefers the refreshed list over the version handed back once the list catches up", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />, {
      searchParams: `?tab=policies&edit_policy=${POLICY_ID}`,
    });
    await screen.findByTestId("flow-builder-target");

    networkingMocks.getPoliciesList.mockResolvedValue({ policies: [{ ...PII_POLICY, version_status: "draft" }] });
    await user.click(screen.getByRole("button", { name: "Publish version" }));

    await waitFor(() => expect(screen.getByTestId("flow-builder-target")).toHaveTextContent(`${POLICY_ID} draft`));
  });

  it("opens the builder for a new policy without writing ?edit_policy=", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<PoliciesPanel accessToken="test-token" userRole="Admin" />, {
      searchParams: "?tab=policies",
      onUrlUpdate,
    });

    await user.click(screen.getByRole("button", { name: "+ Add New Policy" }));
    expect(screen.getByTestId("add-policy-prefill")).toHaveTextContent("empty");
    await user.click(screen.getByRole("button", { name: "Open flow builder" }));

    expect(screen.getByTestId("flow-builder-target")).toHaveTextContent("new policy");
    expect(onUrlUpdate).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Back to policies" }));

    expect(await screen.findByRole("tab", { name: "Policies" })).toHaveAttribute("aria-selected", "true");
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });
});
