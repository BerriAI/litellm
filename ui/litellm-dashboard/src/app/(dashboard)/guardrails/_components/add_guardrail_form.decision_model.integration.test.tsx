import React from "react";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { chooseSelectOption, renderWithProviders } from "@/../tests/test-utils";
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
    await chooseSelectOption(user, await screen.findByLabelText("Decision Provider"), /TypeSafe/);
    await user.click(await screen.findByLabelText("Decision Model"));
    await user.click(await screen.findByTitle("jev-latest"));
  };

  const addQuestion = async (name: string, instructions: string) => {
    fireEvent.click(await screen.findByRole("button", { name: "Add question" }));
    const nameInputs = screen.getAllByLabelText(/^Question \d+ name$/);
    fireEvent.change(nameInputs[nameInputs.length - 1], { target: { value: name } });
    fireEvent.change(screen.getByLabelText(`Question ${nameInputs.length}`), { target: { value: instructions } });
  };

  it("filters the model picker by the selected provider and clears the model when it changes", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    await renderDecisionModel(user);

    await user.click(await screen.findByLabelText("Decision Provider"));
    await user.click(await screen.findByText("TypeSafe"));
    await user.click(await screen.findByLabelText("Decision Model"));
    expect(await screen.findByTitle("jev-latest")).toBeInTheDocument();
    expect(screen.queryByTitle("gpt-5")).not.toBeInTheDocument();
    await user.click(await screen.findByTitle("jev-latest"));

    await user.click(screen.getByLabelText("Decision Provider"));
    await user.click(await screen.findByText("OpenAI"));

    const modelInput = screen.getByLabelText("Decision Model");
    expect(modelInput).not.toHaveValue("jev-latest");
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

    await chooseSelectOption(user, await screen.findByLabelText("Decision Provider"), /TypeSafe/);
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
    await chooseSelectOption(user, await screen.findByLabelText("Decision Provider"), /TypeSafe/);
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
