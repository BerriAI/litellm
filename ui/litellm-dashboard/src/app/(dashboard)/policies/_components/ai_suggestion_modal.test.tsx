import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "@/../tests/test-utils";
import AiSuggestionModal from "./ai_suggestion_modal";

const { suggestPolicyTemplates, modelHubCall, testPolicyTemplate, enrichPolicyTemplateStream } = vi.hoisted(() => ({
  suggestPolicyTemplates: vi.fn(),
  modelHubCall: vi.fn(),
  testPolicyTemplate: vi.fn(),
  enrichPolicyTemplateStream: vi.fn(),
}));

vi.mock("@/components/networking", () => ({
  suggestPolicyTemplates,
  modelHubCall,
  testPolicyTemplate,
  enrichPolicyTemplateStream,
}));

const allTemplates = [
  { id: "tpl-pii", title: "PII Protection", description: "Masks PII", guardrails: ["pii-masker"], complexity: "Low" },
  {
    id: "tpl-inj",
    title: "Injection Defense",
    description: "Blocks prompt injection",
    guardrails: ["prompt-injection"],
    complexity: "Medium",
  },
];

const suggestResponse = {
  selected_templates: [
    { template_id: "tpl-pii", reason: "Your examples contain SSNs" },
    { template_id: "tpl-inj", reason: "Your examples contain instruction overrides" },
  ],
  explanation: "These two cover both risks you described",
};

const defaultProps = {
  visible: true,
  onSelectTemplates: vi.fn(),
  onCancel: vi.fn(),
  accessToken: "sk-test",
  allTemplates,
};

const renderModal = (props: Partial<typeof defaultProps> = {}) =>
  renderWithProviders(<AiSuggestionModal {...defaultProps} {...props} />);

const pickModel = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole("combobox"));
  const options = await screen.findAllByText("gpt-5.1");
  await user.click(options[options.length - 1]);
};

describe("AiSuggestionModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    modelHubCall.mockResolvedValue({ data: [{ model_group: "gpt-5.1" }, { model_group: "claude-opus-4-8" }] });
    suggestPolicyTemplates.mockResolvedValue(suggestResponse);
  });

  it("renders nothing while closed", () => {
    renderModal({ visible: false });

    expect(screen.queryByText("AI Policy Suggestion")).not.toBeInTheDocument();
  });

  it("renders the header and prompt copy when opened", async () => {
    renderModal();

    expect(await screen.findByText("AI Policy Suggestion")).toBeInTheDocument();
    expect(
      screen.getByText("Describe what you want to block and we'll suggest the best policy templates"),
    ).toBeInTheDocument();
  });

  it("loads the model list when opened", async () => {
    renderModal();

    await waitFor(() => {
      expect(modelHubCall).toHaveBeenCalledWith("sk-test");
    });
  });

  it("keeps Suggest disabled until there is both input and a model", async () => {
    const user = userEvent.setup();
    renderModal();

    await screen.findByText("AI Policy Suggestion");
    expect(screen.getByRole("button", { name: "Suggest Policies" })).toBeDisabled();

    await user.type(screen.getByPlaceholderText(/Block PII leakage/), "block PII");
    expect(screen.getByRole("button", { name: "Suggest Policies" })).toBeDisabled();

    await pickModel(user);
    expect(screen.getByRole("button", { name: "Suggest Policies" })).not.toBeDisabled();
  });

  it("sends the examples, description and model to the suggest API", async () => {
    const user = userEvent.setup();
    renderModal();

    await screen.findByText("AI Policy Suggestion");
    await user.type(screen.getByPlaceholderText(/Ignore all previous instructions/), "my ssn is 123");
    await user.type(screen.getByPlaceholderText(/Block PII leakage/), "block PII");
    await pickModel(user);
    await user.click(screen.getByRole("button", { name: "Suggest Policies" }));

    await waitFor(() => {
      expect(suggestPolicyTemplates).toHaveBeenCalledWith("sk-test", ["my ssn is 123"], "block PII", "gpt-5.1");
    });
  });

  it("adds attack example fields up to the maximum of four", async () => {
    const user = userEvent.setup();
    renderModal();

    await screen.findByText("AI Policy Suggestion");
    const countExamples = () => screen.getAllByRole("textbox").length;
    const initial = countExamples();

    await user.click(screen.getByRole("button", { name: "+ Add another example" }));
    expect(countExamples()).toBe(initial + 1);

    await user.click(screen.getByRole("button", { name: "+ Add another example" }));
    await user.click(screen.getByRole("button", { name: "+ Add another example" }));
    expect(countExamples()).toBe(initial + 3);
    expect(screen.queryByRole("button", { name: "+ Add another example" })).not.toBeInTheDocument();
  });

  it("shows each suggested template with the reason it was picked", async () => {
    const user = userEvent.setup();
    renderModal();

    await screen.findByText("AI Policy Suggestion");
    await user.type(screen.getByPlaceholderText(/Block PII leakage/), "block PII");
    await pickModel(user);
    await user.click(screen.getByRole("button", { name: "Suggest Policies" }));

    expect(await screen.findByText("PII Protection")).toBeInTheDocument();
    expect(screen.getByText("Injection Defense")).toBeInTheDocument();
    expect(screen.getByText("Your examples contain SSNs")).toBeInTheDocument();
    expect(screen.getByText("These two cover both risks you described")).toBeInTheDocument();
    expect(screen.getByText("2 templates matched your requirements")).toBeInTheDocument();
  });

  it("preselects every suggestion and reflects the count on the confirm button", async () => {
    const user = userEvent.setup();
    renderModal();

    await screen.findByText("AI Policy Suggestion");
    await user.type(screen.getByPlaceholderText(/Block PII leakage/), "block PII");
    await pickModel(user);
    await user.click(screen.getByRole("button", { name: "Suggest Policies" }));

    expect(await screen.findByRole("button", { name: "Use 2 Selected Templates" })).toBeInTheDocument();
  });

  it("deselecting a suggestion lowers the confirm count", async () => {
    const user = userEvent.setup();
    renderModal();

    await screen.findByText("AI Policy Suggestion");
    await user.type(screen.getByPlaceholderText(/Block PII leakage/), "block PII");
    await pickModel(user);
    await user.click(screen.getByRole("button", { name: "Suggest Policies" }));

    await user.click(await screen.findByText("PII Protection"));

    expect(await screen.findByRole("button", { name: "Use 1 Selected Template" })).toBeInTheDocument();
  });

  it("hands the selected templates back to the caller", async () => {
    const onSelectTemplates = vi.fn();
    const user = userEvent.setup();
    renderModal({ onSelectTemplates });

    await screen.findByText("AI Policy Suggestion");
    await user.type(screen.getByPlaceholderText(/Block PII leakage/), "block PII");
    await pickModel(user);
    await user.click(screen.getByRole("button", { name: "Suggest Policies" }));
    await user.click(await screen.findByRole("button", { name: "Use 2 Selected Templates" }));

    expect(onSelectTemplates).toHaveBeenCalledTimes(1);
    expect(onSelectTemplates.mock.calls[0][0].map((t: { id: string }) => t.id)).toEqual(["tpl-pii", "tpl-inj"]);
  });

  it("returns to the input phase from the results phase", async () => {
    const user = userEvent.setup();
    renderModal();

    await screen.findByText("AI Policy Suggestion");
    await user.type(screen.getByPlaceholderText(/Block PII leakage/), "block PII");
    await pickModel(user);
    await user.click(screen.getByRole("button", { name: "Suggest Policies" }));
    await user.click(await screen.findByRole("button", { name: "Back" }));

    expect(
      await screen.findByText("Describe what you want to block and we'll suggest the best policy templates"),
    ).toBeInTheDocument();
  });

  it("reports an empty result set instead of failing silently", async () => {
    suggestPolicyTemplates.mockRejectedValue(new Error("boom"));
    const user = userEvent.setup();
    renderModal();

    await screen.findByText("AI Policy Suggestion");
    await user.type(screen.getByPlaceholderText(/Block PII leakage/), "block PII");
    await pickModel(user);
    await user.click(screen.getByRole("button", { name: "Suggest Policies" }));

    expect(await screen.findByText("No matching templates found")).toBeInTheDocument();
    expect(screen.getByText("Try adjusting your examples or description.")).toBeInTheDocument();
  });

  it("cancels back to the caller", async () => {
    const onCancel = vi.fn();
    const user = userEvent.setup();
    renderModal({ onCancel });

    await screen.findByText("AI Policy Suggestion");
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(onCancel).toHaveBeenCalledTimes(1);
  });
});
