import React from "react";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "@/../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  createGuardrailCall,
  decisionsTestCall,
  getGuardrailProviderSpecificParams,
  getGuardrailUISettings,
  modelHubCall,
} from "@/components/networking";
import { toast } from "@/lib/toast";
import AddGuardrailForm from "./add_guardrail_form";
import { GUARDRAIL_PRESETS } from "./guardrail_garden_configs";

vi.mock("@/lib/toast", () => ({
  toast: {
    success: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    fromError: vi.fn(),
    dismiss: vi.fn(),
  },
}));

vi.mock("@/components/networking", () => ({
  createGuardrailCall: vi.fn(),
  decisionsTestCall: vi.fn(),
  getGuardrailProviderSpecificParams: vi.fn().mockResolvedValue({}),
  getGuardrailUISettings: vi.fn().mockResolvedValue({}),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
  modelHubCall: vi.fn().mockResolvedValue({ data: [] }),
}));

describe("AddGuardrailForm decision model questions", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
    vi.mocked(getGuardrailProviderSpecificParams).mockResolvedValue({
      decision_model: { ui_friendly_name: "Decision Model" },
    });
    vi.mocked(getGuardrailUISettings).mockResolvedValue({
      decision_model_providers: ["typesafe", "openai"],
    });
    vi.mocked(modelHubCall).mockResolvedValue({
      data: [
        { model_group: "jev-latest", providers: ["typesafe"], mode: "evaluation" },
        { model_group: "gpt-5", providers: ["openai"], mode: "evaluation" },
      ],
    });
  });

  const renderDecisionModel = async (user: ReturnType<typeof userEvent.setup>) => {
    renderWithProviders(
      <AddGuardrailForm visible={true} onClose={vi.fn()} accessToken="test-token" onSuccess={vi.fn()} />,
    );
    fireEvent.change(await screen.findByLabelText("Guardrail Name"), { target: { value: "dm-1" } });
    await user.click(screen.getByLabelText("Guardrail Provider"));
    await user.click(await screen.findByText("Decision Model"));
    await user.click(screen.getByRole("button", { name: "Next" }));
  };

  const pickTypesafeModel = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(await screen.findByLabelText("Decision Model"));
    await user.click(await screen.findByTitle("jev-latest"));
  };

  const addQuestion = async (name: string, instructions: string) => {
    fireEvent.click(await screen.findByRole("button", { name: "Add question" }));
    const nameInputs = screen.getAllByLabelText(/^Question \d+ name$/);
    fireEvent.change(nameInputs[nameInputs.length - 1], { target: { value: name } });
    fireEvent.change(screen.getByLabelText(`Question ${nameInputs.length}`), { target: { value: instructions } });
  };

  it("lists every decision model on the proxy with its provider logo and no provider picker", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    vi.mocked(modelHubCall).mockResolvedValue({
      data: [
        { model_group: "jev-latest", providers: ["typesafe"], mode: "evaluation" },
        { model_group: "gpt-5", providers: ["openai"], mode: "evaluation" },
        { model_group: "gpt-4o-mini", providers: ["openai"], mode: "chat" },
      ],
    });
    await renderDecisionModel(user);

    await user.click(await screen.findByLabelText("Decision Model"));
    const jev = await screen.findByTitle("jev-latest");
    expect(within(jev).getByRole("img", { name: "TypeSafe logo" })).toBeInTheDocument();
    expect(within(screen.getByTitle("gpt-5")).getByRole("img", { name: "OpenAI logo" })).toBeInTheDocument();
    expect(screen.queryByTitle("gpt-4o-mini")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Decision Provider")).not.toBeInTheDocument();
    expect(screen.queryByRole("note", { name: "No decision models" })).not.toBeInTheDocument();

    await user.click(jev);
    expect(screen.getByLabelText("Decision Model")).toHaveValue("jev-latest");
  });

  it("prompts to add a model when the proxy has no decision models", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    vi.mocked(modelHubCall).mockResolvedValue({
      data: [{ model_group: "gpt-6-luna", providers: ["openai"], mode: "chat" }],
    });
    await renderDecisionModel(user);

    const note = await screen.findByRole("note", { name: "No decision models" });
    expect(note).toHaveTextContent("Add a decision model to use this guardrail");
    expect(within(note).getByRole("link", { name: "Add a model" })).toHaveAttribute(
      "href",
      expect.stringMatching(/\/models-and-endpoints$/),
    );
    expect(screen.getByLabelText("Decision Model")).toHaveAttribute("placeholder", "No decision models yet");
  });

  const renderFromCard = (visible: boolean, accessToken = "test-token") => (
    <AddGuardrailForm
      visible={visible}
      onClose={vi.fn()}
      accessToken={accessToken}
      onSuccess={vi.fn()}
      preset={GUARDRAIL_PRESETS.dm_typesafe_jev}
    />
  );

  it("prefills the garden card's provider model and lists only that provider's models", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    vi.mocked(modelHubCall).mockResolvedValue({
      data: [
        { model_group: "jev-preview", providers: ["typesafe"], mode: "evaluation" },
        { model_group: "gpt-5", providers: ["openai"], mode: "evaluation" },
        { model_group: "jev-latest", providers: ["typesafe"], mode: "evaluation" },
      ],
    });
    renderWithProviders(renderFromCard(true));
    await waitFor(() => expect(screen.getByLabelText("Guardrail Name")).toHaveValue("TypeSafe Jev"));
    await user.click(screen.getByRole("button", { name: "Next" }));

    expect(await screen.findByLabelText("Decision Model")).toHaveValue("jev-latest");
    await user.click(screen.getByLabelText("Decision Model"));
    expect(await screen.findByTitle("jev-preview")).toBeInTheDocument();
    expect(screen.queryByTitle("gpt-5")).not.toBeInTheDocument();
  });

  it("prefills the card's provider model again after the form is closed and reopened", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    const { rerender } = renderWithProviders(renderFromCard(true));
    await waitFor(() => expect(screen.getByLabelText("Guardrail Name")).toHaveValue("TypeSafe Jev"));
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(await screen.findByLabelText("Decision Model")).toHaveValue("jev-latest");

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    rerender(renderFromCard(false));
    rerender(renderFromCard(true));
    await waitFor(() => expect(screen.getByLabelText("Guardrail Name")).toHaveValue("TypeSafe Jev"));
    await user.click(screen.getByRole("button", { name: "Next" }));

    expect(await screen.findByLabelText("Decision Model")).toHaveValue("jev-latest");
  });

  it("keeps a model the user picked when the model list reloads", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    vi.mocked(modelHubCall).mockResolvedValue({
      data: [
        { model_group: "jev-latest", providers: ["typesafe"], mode: "evaluation" },
        { model_group: "jev-preview", providers: ["typesafe"], mode: "evaluation" },
      ],
    });
    const { rerender } = renderWithProviders(renderFromCard(true));
    await waitFor(() => expect(screen.getByLabelText("Guardrail Name")).toHaveValue("TypeSafe Jev"));
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(await screen.findByLabelText("Decision Model")).toHaveValue("jev-latest");
    await user.click(screen.getByLabelText("Decision Model"));
    await user.click(await screen.findByTitle("jev-preview"));

    vi.mocked(modelHubCall).mockResolvedValue({
      data: [
        { model_group: "jev-1.13.0", providers: ["typesafe"], mode: "evaluation" },
        { model_group: "jev-latest", providers: ["typesafe"], mode: "evaluation" },
        { model_group: "jev-preview", providers: ["typesafe"], mode: "evaluation" },
      ],
    });
    rerender(renderFromCard(true, "refreshed-token"));
    await user.click(screen.getByLabelText("Decision Model"));
    expect(await screen.findByTitle("jev-1.13.0")).toBeInTheDocument();
    expect(screen.getByLabelText("Decision Model")).toHaveValue("jev-preview");
  });

  it("leaves the model empty when the form is not opened from a card", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    await renderDecisionModel(user);

    expect(await screen.findByLabelText("Decision Model")).toHaveValue("");
  });

  it("names the card's provider when it has no decision models", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    renderWithProviders(
      <AddGuardrailForm
        visible={true}
        onClose={vi.fn()}
        accessToken="test-token"
        onSuccess={vi.fn()}
        preset={GUARDRAIL_PRESETS.dm_databricks}
      />,
    );
    await waitFor(() => expect(screen.getByLabelText("Guardrail Name")).toHaveValue("Databricks Decision Model"));
    await user.click(screen.getByRole("button", { name: "Next" }));

    expect(await screen.findByRole("note", { name: "No decision models" })).toHaveTextContent(
      "Add a decision model from the Databricks (Qwen API) provider to use this guardrail",
    );
  });

  it("shows only an Add question button until it is clicked, then a blank question per click", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    await renderDecisionModel(user);
    await pickTypesafeModel(user);

    expect(screen.queryByLabelText(/^Question \d+ name$/)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Add question" }));
    expect(screen.getByLabelText("Question 1 name")).toHaveValue("");
    expect(screen.getByLabelText("Question 1")).toHaveValue("");
    expect(screen.getByText("0.70")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Add question" }));
    expect(screen.getByLabelText("Question 2 name")).toHaveValue("");
  });

  it("submits a question with the chosen action and the default threshold", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    await renderDecisionModel(user);
    await pickTypesafeModel(user);
    await addQuestion("invoice_policy", "Does the text ask about invoices?");

    await user.click(screen.getByLabelText("Question 1 action"));
    await user.click(await screen.findByText("Log only"));

    fireEvent.click(screen.getByRole("button", { name: "Create Guardrail" }));

    await waitFor(() => expect(vi.mocked(createGuardrailCall)).toHaveBeenCalled());
    const submitted = vi.mocked(createGuardrailCall).mock.calls[0][1];
    expect(submitted.litellm_params.checks).toEqual([
      {
        name: "invoice_policy",
        instructions: "Does the text ask about invoices?",
        action: "log",
        threshold: 0.7,
      },
    ]);
  });

  it("submits a question edited after another was added", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    await renderDecisionModel(user);
    await pickTypesafeModel(user);
    await addQuestion("invoice_policy", "Does the text ask about invoices?");
    await addQuestion("refund_policy", "Is this about refunds?");

    fireEvent.change(screen.getByLabelText("Question 1 name"), { target: { value: "billing_policy" } });
    fireEvent.change(screen.getByLabelText("Question 1"), { target: { value: "Is this about billing?" } });
    fireEvent.click(screen.getByRole("button", { name: "Create Guardrail" }));

    await waitFor(() => expect(vi.mocked(createGuardrailCall)).toHaveBeenCalled());
    const submitted = vi.mocked(createGuardrailCall).mock.calls[0][1];
    expect(
      submitted.litellm_params.checks.map((c: { name: string; instructions: string }) => [c.name, c.instructions]),
    ).toEqual([
      ["billing_policy", "Is this about billing?"],
      ["refund_policy", "Is this about refunds?"],
    ]);
  });

  it("flags two questions with the same name and blocks create", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    await renderDecisionModel(user);
    await pickTypesafeModel(user);
    await addQuestion("invoice_policy", "Does the text ask about invoices?");
    await addQuestion("invoice_policy", "Duplicate name");

    expect(screen.getAllByText("Another question already uses this name")).toHaveLength(2);

    fireEvent.click(screen.getByRole("button", { name: "Create Guardrail" }));
    await waitFor(() =>
      expect(vi.mocked(toast.fromError)).toHaveBeenCalledWith("Each question needs a different name"),
    );
    expect(vi.mocked(createGuardrailCall)).not.toHaveBeenCalled();
  });

  it("blocks create while a question has no text", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    await renderDecisionModel(user);
    await pickTypesafeModel(user);
    await addQuestion("invoice_policy", "Does the text ask about invoices?");
    fireEvent.click(screen.getByRole("button", { name: "Add question" }));
    fireEvent.change(screen.getByLabelText("Question 2 name"), { target: { value: "refund_policy" } });

    fireEvent.click(screen.getByRole("button", { name: "Create Guardrail" }));
    await waitFor(() =>
      expect(vi.mocked(toast.fromError)).toHaveBeenCalledWith(
        "Give every question a name and a question, or remove it",
      ),
    );
    expect(vi.mocked(createGuardrailCall)).not.toHaveBeenCalled();
  });

  it("removes a question via its Remove button", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    await renderDecisionModel(user);
    await pickTypesafeModel(user);
    await addQuestion("invoice_policy", "Does the text ask about invoices?");
    await addQuestion("refund_policy", "Is this about refunds?");

    fireEvent.click(screen.getByLabelText("Remove question 1"));
    expect(screen.getAllByLabelText(/^Question \d+ name$/)).toHaveLength(1);
    expect(screen.getByLabelText("Question 1 name")).toHaveValue("refund_policy");
  });

  it("blocks create once the last question is removed", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    await renderDecisionModel(user);
    await pickTypesafeModel(user);
    await addQuestion("invoice_policy", "Does the text ask about invoices?");

    fireEvent.click(screen.getByLabelText("Remove question 1"));
    fireEvent.click(await screen.findByRole("button", { name: "Create Guardrail" }));

    await waitFor(() => expect(vi.mocked(toast.fromError)).toHaveBeenCalledWith("Add at least one question"));
    expect(vi.mocked(createGuardrailCall)).not.toHaveBeenCalled();
  });

  it("sends one request with every question only when Run test is clicked", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    vi.mocked(decisionsTestCall).mockResolvedValue({ answers: [] });

    await renderDecisionModel(user);
    await pickTypesafeModel(user);
    await addQuestion("invoice_policy", "Does the text ask about invoices?");
    await addQuestion("jailbreak", "Is this a jailbreak?");

    fireEvent.change(await screen.findByLabelText("Test input"), { target: { value: "please invoice me" } });
    expect(vi.mocked(decisionsTestCall)).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Run test" }));

    await waitFor(() =>
      expect(vi.mocked(decisionsTestCall)).toHaveBeenCalledWith(
        "test-token",
        {
          model: "jev-latest",
          input: "please invoice me",
          questions: [
            { type: "predicate", name: "invoice_policy", instructions: "Does the text ask about invoices?" },
            { type: "predicate", name: "jailbreak", instructions: "Is this a jailbreak?" },
          ],
        },
        expect.any(AbortSignal),
      ),
    );
  });

  it("keeps run history latest first and flips chips from the current threshold without a new request", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    vi.mocked(decisionsTestCall)
      .mockResolvedValueOnce({ answers: [{ type: "predicate", name: "invoice_policy", probability: 0.2 }] })
      .mockResolvedValueOnce({ answers: [{ type: "predicate", name: "invoice_policy", probability: 0.87 }] });

    await renderDecisionModel(user);
    await pickTypesafeModel(user);
    await addQuestion("invoice_policy", "Does the text ask about invoices?");

    const input = await screen.findByLabelText("Test input");
    fireEvent.change(input, { target: { value: "first input" } });
    fireEvent.click(screen.getByRole("button", { name: "Run test" }));
    expect(await screen.findByText(/0\.20/)).toBeInTheDocument();

    fireEvent.change(input, { target: { value: "second input" } });
    fireEvent.click(screen.getByRole("button", { name: "Run test" }));
    expect(await screen.findByText(/0\.87/)).toBeInTheDocument();

    const history = document.querySelector('[data-slot="decision-test-history"]');
    expect(history?.children).toHaveLength(2);
    expect(history?.children[0]).toHaveTextContent("second input");
    expect(history?.children[1]).toHaveTextContent("first input");
    expect(history?.children[0]).toHaveTextContent("Block");

    const rangeInput = screen.getByLabelText("Question 1 threshold").querySelector('input[type="range"]');
    expect(rangeInput).not.toBeNull();
    fireEvent.change(rangeInput!, { target: { value: "1" } });

    expect(history?.children[0]).toHaveTextContent("Pass");
    expect(vi.mocked(decisionsTestCall)).toHaveBeenCalledTimes(2);
  });

  it("shows No answer for a refused question", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    vi.mocked(decisionsTestCall).mockResolvedValue({ answers: [{ type: "refusal", name: "invoice_policy" }] });

    await renderDecisionModel(user);
    await pickTypesafeModel(user);
    await addQuestion("invoice_policy", "Does the text ask about invoices?");

    fireEvent.change(await screen.findByLabelText("Test input"), { target: { value: "x" } });
    fireEvent.click(screen.getByRole("button", { name: "Run test" }));

    expect(await screen.findByText(/No answer/)).toBeInTheDocument();
  });
});

describe("AddGuardrailForm provider search", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("finds Decision Model by search aliases and keeps label search working", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    vi.mocked(getGuardrailProviderSpecificParams).mockResolvedValue({
      decision_model: { ui_friendly_name: "Decision Model" },
    });
    renderWithProviders(
      <AddGuardrailForm visible={true} onClose={vi.fn()} accessToken="test-token" onSuccess={vi.fn()} />,
    );

    const provider = await screen.findByLabelText("Guardrail Provider");
    await user.click(provider);

    await user.type(provider, "jev");
    expect(await screen.findByText("Decision Model")).toBeInTheDocument();

    await user.clear(provider);
    await user.type(provider, "prompt injection");
    expect(await screen.findByText("Decision Model")).toBeInTheDocument();

    await user.clear(provider);
    await user.type(provider, "presidio");
    expect(await screen.findByText("Presidio PII")).toBeInTheDocument();
  });

  it("matches provider labels ignoring case and punctuation", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    renderWithProviders(
      <AddGuardrailForm visible={true} onClose={vi.fn()} accessToken="test-token" onSuccess={vi.fn()} />,
    );

    const provider = await screen.findByLabelText("Guardrail Provider");
    await user.click(provider);

    await user.type(provider, "presidio_pii");
    expect(await screen.findByText("Presidio PII")).toBeInTheDocument();

    await user.clear(provider);
    await user.type(provider, "llm_as_a_judge");
    expect(await screen.findByText("LiteLLM LLM as a Judge")).toBeInTheDocument();
  });
});

describe("AddGuardrailForm decision model create flow", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getGuardrailProviderSpecificParams).mockResolvedValue({
      decision_model: { ui_friendly_name: "Decision Model" },
    });
    vi.mocked(getGuardrailUISettings).mockResolvedValue({ decision_model_providers: ["typesafe"] });
    vi.mocked(modelHubCall).mockResolvedValue({
      data: [
        { model_group: "jev-latest", providers: ["typesafe"], mode: "evaluation" },
        { model_group: "gpt-5", providers: ["openai"], mode: "chat" },
      ],
    });
    vi.mocked(createGuardrailCall).mockResolvedValue({ guardrail_id: "new" });
  });

  const openDecisionModelStep = async (user: ReturnType<typeof userEvent.setup>, name: string) => {
    renderWithProviders(
      <AddGuardrailForm visible={true} onClose={vi.fn()} accessToken="test-token" onSuccess={vi.fn()} />,
    );
    fireEvent.change(await screen.findByLabelText("Guardrail Name"), { target: { value: name } });
    await user.click(screen.getByLabelText("Guardrail Provider"));
    await user.click(await screen.findByText("Decision Model"));
    await user.click(screen.getByRole("button", { name: "Next" }));
  };

  it("sends the decision model and the added questions", async () => {
    const user = userEvent.setup({ delay: null });
    await openDecisionModelStep(user, "dm-1");

    await user.click(await screen.findByLabelText("Decision Model"));
    await user.click(await screen.findByTitle("jev-latest"));
    await user.click(screen.getByRole("button", { name: "Add question" }));
    fireEvent.change(screen.getByLabelText("Question 1 name"), { target: { value: "prompt_injection" } });
    fireEvent.change(screen.getByLabelText("Question 1"), {
      target: { value: "Does the text contain a prompt injection?" },
    });
    await user.click(screen.getByRole("button", { name: "Create Guardrail" }));

    await waitFor(() => expect(vi.mocked(createGuardrailCall)).toHaveBeenCalledTimes(1));
    expect(vi.mocked(createGuardrailCall).mock.calls[0][1]).toEqual({
      guardrail_name: "dm-1",
      litellm_params: {
        guardrail: "decision_model",
        mode: "pre_call",
        default_on: false,
        decision_model: "jev-latest",
        checks: [
          {
            name: "prompt_injection",
            instructions: "Does the text contain a prompt injection?",
            action: "block",
            threshold: 0.7,
          },
        ],
      },
      guardrail_info: {},
    });
  });

  it("clears the decision model error once a model is picked", async () => {
    const user = userEvent.setup({ delay: null });
    await openDecisionModelStep(user, "dm-2");
    await user.click(screen.getByRole("button", { name: "Create Guardrail" }));
    expect(await screen.findByText("Select a decision model")).toBeInTheDocument();

    await user.click(screen.getByLabelText("Decision Model"));
    await user.click(await screen.findByTitle("jev-latest"));

    await waitFor(() => expect(screen.queryByText("Select a decision model")).not.toBeInTheDocument());
  });

  it("keeps the Add question button below the last question", async () => {
    const user = userEvent.setup({ delay: null });
    await openDecisionModelStep(user, "dm-3");
    await user.click(await screen.findByRole("button", { name: "Add question" }));
    await user.click(screen.getByRole("button", { name: "Add question" }));

    const lastQuestion = screen.getByLabelText("Question 2");
    const addButton = screen.getByRole("button", { name: "Add question" });
    expect(lastQuestion.compareDocumentPosition(addButton) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
});
