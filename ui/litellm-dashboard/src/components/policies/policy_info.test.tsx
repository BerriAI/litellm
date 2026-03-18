import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "../networking";
import PolicyInfoView from "./policy_info";
import { Policy } from "./types";

vi.mock("../networking");
vi.mock("./pipeline_flow_builder", () => ({
  PipelineInfoDisplay: () => <div data-testid="pipeline-info" />,
}));

const basePolicy: Policy = {
  policy_id: "policy-uuid-1",
  policy_name: "My Test Policy",
  inherit: null,
  description: "A test description",
  guardrails_add: ["guardrail-a"],
  guardrails_remove: [],
  condition: null,
};

const defaultProps = {
  policyId: "policy-uuid-1",
  onClose: vi.fn(),
  onEdit: vi.fn(),
  accessToken: "test-token",
  isAdmin: true,
  getPolicy: vi.fn(),
};

describe("PolicyInfoView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should not show policy content while the fetch is in flight", () => {
    defaultProps.getPolicy.mockReturnValue(new Promise(() => {}));
    vi.mocked(networking.getResolvedGuardrails).mockReturnValue(new Promise(() => {}));
    renderWithProviders(<PolicyInfoView {...defaultProps} />);
    expect(screen.queryByText("My Test Policy")).not.toBeInTheDocument();
  });

  it("should show a 'Policy not found' message when getPolicy resolves null", async () => {
    defaultProps.getPolicy.mockResolvedValue(null);
    vi.mocked(networking.getResolvedGuardrails).mockResolvedValue({ resolved_guardrails: [] });
    renderWithProviders(<PolicyInfoView {...defaultProps} />);
    expect(await screen.findByText(/policy not found/i)).toBeInTheDocument();
  });

  it("should render the policy name after loading", async () => {
    defaultProps.getPolicy.mockResolvedValue(basePolicy);
    vi.mocked(networking.getResolvedGuardrails).mockResolvedValue({ resolved_guardrails: [] });
    renderWithProviders(<PolicyInfoView {...defaultProps} />);
    expect(await screen.findByText("My Test Policy")).toBeInTheDocument();
  });

  it("should render the policy ID", async () => {
    defaultProps.getPolicy.mockResolvedValue(basePolicy);
    vi.mocked(networking.getResolvedGuardrails).mockResolvedValue({ resolved_guardrails: [] });
    renderWithProviders(<PolicyInfoView {...defaultProps} />);
    expect(await screen.findByText("policy-uuid-1")).toBeInTheDocument();
  });

  it("should render guardrails_add tags", async () => {
    defaultProps.getPolicy.mockResolvedValue(basePolicy);
    vi.mocked(networking.getResolvedGuardrails).mockResolvedValue({ resolved_guardrails: [] });
    renderWithProviders(<PolicyInfoView {...defaultProps} />);
    expect(await screen.findByText("guardrail-a")).toBeInTheDocument();
  });

  it("should call onClose when the Back to Policies button is clicked", async () => {
    defaultProps.getPolicy.mockResolvedValue(basePolicy);
    vi.mocked(networking.getResolvedGuardrails).mockResolvedValue({ resolved_guardrails: [] });
    const user = userEvent.setup();
    renderWithProviders(<PolicyInfoView {...defaultProps} />);
    await screen.findByText("My Test Policy");
    await user.click(screen.getByRole("button", { name: /back to policies/i }));
    expect(defaultProps.onClose).toHaveBeenCalled();
  });

  it("should call onEdit with the policy when the Edit Policy button is clicked", async () => {
    defaultProps.getPolicy.mockResolvedValue(basePolicy);
    vi.mocked(networking.getResolvedGuardrails).mockResolvedValue({ resolved_guardrails: [] });
    const user = userEvent.setup();
    renderWithProviders(<PolicyInfoView {...defaultProps} isAdmin />);
    await screen.findByText("My Test Policy");
    await user.click(screen.getByRole("button", { name: /edit policy/i }));
    expect(defaultProps.onEdit).toHaveBeenCalledWith(basePolicy);
  });

  it("should not show the Edit Policy button for non-admins", async () => {
    defaultProps.getPolicy.mockResolvedValue(basePolicy);
    vi.mocked(networking.getResolvedGuardrails).mockResolvedValue({ resolved_guardrails: [] });
    renderWithProviders(<PolicyInfoView {...defaultProps} isAdmin={false} />);
    await screen.findByText("My Test Policy");
    expect(screen.queryByRole("button", { name: /edit policy/i })).not.toBeInTheDocument();
  });

  it("should display resolved guardrails when returned from the API", async () => {
    defaultProps.getPolicy.mockResolvedValue(basePolicy);
    vi.mocked(networking.getResolvedGuardrails).mockResolvedValue({
      resolved_guardrails: ["resolved-guardrail-x"],
    });
    renderWithProviders(<PolicyInfoView {...defaultProps} />);
    expect(await screen.findByText("resolved-guardrail-x")).toBeInTheDocument();
  });

  it("should display the model condition tag when present", async () => {
    const policyWithCondition = { ...basePolicy, condition: { model: "gpt-4" } };
    defaultProps.getPolicy.mockResolvedValue(policyWithCondition);
    vi.mocked(networking.getResolvedGuardrails).mockResolvedValue({ resolved_guardrails: [] });
    renderWithProviders(<PolicyInfoView {...defaultProps} />);
    expect(await screen.findByText("gpt-4")).toBeInTheDocument();
  });

  it("should show 'No model condition' when condition is null", async () => {
    defaultProps.getPolicy.mockResolvedValue(basePolicy);
    vi.mocked(networking.getResolvedGuardrails).mockResolvedValue({ resolved_guardrails: [] });
    renderWithProviders(<PolicyInfoView {...defaultProps} />);
    expect(await screen.findByText(/no model condition/i)).toBeInTheDocument();
  });

  it("should show the formatted created_at date", async () => {
    const policy = { ...basePolicy, created_at: "2024-06-15T12:00:00Z" };
    defaultProps.getPolicy.mockResolvedValue(policy);
    vi.mocked(networking.getResolvedGuardrails).mockResolvedValue({ resolved_guardrails: [] });
    renderWithProviders(<PolicyInfoView {...defaultProps} />);
    await waitFor(() => {
      expect(screen.getByText(/2024-06-15/)).toBeInTheDocument();
    });
  });
});
