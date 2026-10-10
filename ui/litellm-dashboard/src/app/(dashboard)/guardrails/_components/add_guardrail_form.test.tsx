import React from "react";
import { fireEvent, screen, waitFor } from "@testing-library/react";
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

const renderForm = () => {
  const onClose = vi.fn();
  renderWithProviders(<AddGuardrailForm visible={true} onClose={onClose} accessToken={null} onSuccess={vi.fn()} />);
  return { onClose };
};

describe("AddGuardrailForm close behavior", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("does not close when the user clicks outside the modal on the mask", () => {
    const { onClose } = renderForm();
    expect(screen.getByText("Create guardrail")).toBeInTheDocument();

    const backdrop = document.querySelector('[data-slot="dialog-overlay"]') as HTMLElement;
    expect(backdrop).toBeTruthy();
    fireEvent.mouseDown(backdrop);
    fireEvent.mouseUp(backdrop);
    fireEvent.click(backdrop);

    expect(onClose).not.toHaveBeenCalled();
  });

  it("closes when the user clicks the explicit close button", () => {
    const { onClose } = renderForm();
    fireEvent.click(screen.getByRole("button", { name: "✕" }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("AddGuardrailForm provider options", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders provider options with logos from the bundled guardrail logo map", async () => {
    renderForm();
    fireEvent.mouseDown(screen.getByLabelText("Guardrail Provider"));

    const logo = await screen.findByAltText("Presidio PII logo");
    expect(logo).toHaveAttribute("src", expect.stringContaining("microsoft_azure.svg"));
  });
});

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
        { model_group: "jev-latest", providers: ["typesafe"] },
        { model_group: "gpt-5", providers: ["openai"] },
      ],
    });
  });

  const renderDecisionModel = async (user: ReturnType<typeof userEvent.setup>) => {
    renderWithProviders(
      <AddGuardrailForm visible={true} onClose={vi.fn()} accessToken="test-token" onSuccess={vi.fn()} />,
    );
    await user.type(await screen.findByLabelText("Guardrail Name"), "dm-1");
    await user.click(screen.getByLabelText("Guardrail Provider"));
    await user.click(await screen.findByText("Decision Model"));
    await user.click(screen.getByRole("button", { name: "Next" }));
  };

  const pickTypesafeModel = async (user: ReturnType<typeof userEvent.setup>) => {
    await user.click(await screen.findByLabelText("Decision Provider"));
    await user.click(await screen.findByText("TypeSafe"));
    await user.click(await screen.findByLabelText("Decision Model"));
    await user.click(await screen.findByTitle("jev-latest"));
  };

  const addQuestion = async (user: ReturnType<typeof userEvent.setup>, name: string, instructions: string) => {
    await user.type(await screen.findByLabelText("Question name"), name);
    await user.type(screen.getByLabelText("Question"), instructions);
    fireEvent.click(screen.getByRole("button", { name: "Add question" }));
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

  it("adds a question with the chosen action and threshold and submits its instructions", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    await renderDecisionModel(user);
    await pickTypesafeModel(user);

    expect(screen.getByText("0.70")).toBeInTheDocument();

    await user.type(await screen.findByLabelText("Question name"), "invoice_policy");
    await user.type(screen.getByLabelText("Question"), "Does the text ask about invoices?");
    await user.click(screen.getByLabelText("New question action"));
    await user.click(await screen.findByText("Log only"));
    fireEvent.click(screen.getByRole("button", { name: "Add question" }));

    const checkbox = await screen.findByRole("checkbox", { name: "invoice_policy" });
    expect(checkbox).toBeChecked();
    expect(screen.getAllByText("0.70")).toHaveLength(2);

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

  it("drops an unchecked question from the submitted params", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    await renderDecisionModel(user);
    await pickTypesafeModel(user);
    await addQuestion(user, "invoice_policy", "Does the text ask about invoices?");

    fireEvent.click(await screen.findByRole("checkbox", { name: "invoice_policy" }));
    expect(screen.getByLabelText("invoice_policy action")).toBeDisabled();
    expect(screen.getByLabelText("invoice_policy threshold")).toHaveAttribute("data-disabled");

    await user.type(await screen.findByLabelText("Question name"), "refund_policy");
    await user.type(screen.getByLabelText("Question"), "Is this about refunds?");
    fireEvent.click(screen.getByRole("button", { name: "Add question" }));

    fireEvent.click(await screen.findByRole("button", { name: "Create Guardrail" }));

    await waitFor(() => expect(vi.mocked(createGuardrailCall)).toHaveBeenCalled());
    const submitted = vi.mocked(createGuardrailCall).mock.calls[0][1];
    const names = submitted.litellm_params.checks.map((c: { name: string }) => c.name);
    expect(names).toEqual(["refund_policy"]);
  });

  it("rejects a second question with a name already used", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    await renderDecisionModel(user);
    await pickTypesafeModel(user);
    await addQuestion(user, "invoice_policy", "Does the text ask about invoices?");
    await addQuestion(user, "invoice_policy", "Duplicate name");

    expect(await screen.findByText("That name is already used by another question")).toBeInTheDocument();
    expect(screen.getAllByRole("checkbox", { name: "invoice_policy" })).toHaveLength(1);
  });

  it("removes a question row via its Remove button", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    await renderDecisionModel(user);
    await pickTypesafeModel(user);
    await addQuestion(user, "invoice_policy", "Does the text ask about invoices?");

    fireEvent.click(await screen.findByLabelText("Remove invoice_policy"));
    expect(screen.queryByRole("checkbox", { name: "invoice_policy" })).not.toBeInTheDocument();
  });

  it("blocks create when the only question left is unchecked", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    await renderDecisionModel(user);
    await pickTypesafeModel(user);
    await addQuestion(user, "invoice_policy", "Does the text ask about invoices?");

    fireEvent.click(screen.getByRole("checkbox", { name: "invoice_policy" }));
    fireEvent.click(await screen.findByRole("button", { name: "Create Guardrail" }));

    await waitFor(() => expect(vi.mocked(toast.fromError)).toHaveBeenCalledWith("Add at least one question"));
    expect(vi.mocked(createGuardrailCall)).not.toHaveBeenCalled();
  });

  it("sends one request with every enabled question only when Run test is clicked", async () => {
    const user = userEvent.setup({ delay: null, pointerEventsCheck: 0 });
    vi.mocked(decisionsTestCall).mockResolvedValue({ answers: [] });

    await renderDecisionModel(user);
    await pickTypesafeModel(user);
    await addQuestion(user, "invoice_policy", "Does the text ask about invoices?");
    await addQuestion(user, "jailbreak", "Is this a jailbreak?");

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
    await addQuestion(user, "invoice_policy", "Does the text ask about invoices?");

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

    const rangeInput = screen.getByLabelText("invoice_policy threshold").querySelector('input[type="range"]');
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
    await addQuestion(user, "invoice_policy", "Does the text ask about invoices?");

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
});
