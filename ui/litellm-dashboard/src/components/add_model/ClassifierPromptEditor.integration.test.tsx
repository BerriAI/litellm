import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import ClassifierPromptEditor from "./ClassifierPromptEditor";

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test" }),
}));

const getDefaultPrompt = vi.hoisted(() => vi.fn());
vi.mock("@/components/networking", () => ({
  getAutoRouterClassifierDefaultPromptCall: getDefaultPrompt,
}));

const DEFAULT_PROMPT = "Classify the complexity of a user request into exactly one tier. Tiers: SIMPLE ...";

beforeEach(() => {
  getDefaultPrompt.mockReset();
  getDefaultPrompt.mockResolvedValue(DEFAULT_PROMPT);
});

const openEditor = async (
  systemPrompt?: string,
  onChange = vi.fn(),
  contextWindowSize = 3,
  tierLabels?: Record<string, string>,
) => {
  renderWithProviders(
    <ClassifierPromptEditor
      systemPrompt={systemPrompt}
      onChange={onChange}
      contextWindowSize={contextWindowSize}
      tierLabels={tierLabels}
    />,
  );
  await userEvent.click(screen.getByRole("button", { name: /prompt/i }));
  await waitFor(() => expect(screen.getByLabelText("Classifier system prompt")).toBeInTheDocument());
  return onChange;
};

describe("ClassifierPromptEditor", () => {
  it("prefills the live rubric fetched for the configured context window", async () => {
    await openEditor(undefined, vi.fn(), 7);
    // Prefilling from the backend rather than a frontend copy is the whole point: a copy would
    // drift the moment the rubric is edited.
    expect(getDefaultPrompt).toHaveBeenCalledWith("sk-test", 7, undefined);
    expect(screen.getByLabelText("Classifier system prompt")).toHaveValue(DEFAULT_PROMPT);
  });

  it("prefills the rubric named by the operator's renamed tiers", async () => {
    // A renamed router sends a rubric using its own labels, and its classifier must return them,
    // so prefilling the canonical names would hand back a prompt that router rejects.
    const tierLabels = { SIMPLE: "Cheap", REASONING: "Deep" };
    await openEditor(undefined, vi.fn(), 7, tierLabels);
    expect(getDefaultPrompt).toHaveBeenCalledWith("sk-test", 7, tierLabels);
  });

  it("warns that the prompt replaces the injection-defense text", async () => {
    await openEditor();
    expect(screen.getByText("Proceed with caution")).toBeInTheDocument();
    expect(screen.getByText(/entire system role/)).toBeInTheDocument();
  });

  it("saves an edited prompt as an override", async () => {
    const onChange = await openEditor();
    const textarea = screen.getByLabelText("Classifier system prompt");
    await userEvent.clear(textarea);
    await userEvent.type(textarea, "Grade data sensitivity");
    await userEvent.click(screen.getByRole("button", { name: "Save prompt" }));
    expect(onChange).toHaveBeenCalledWith("Grade data sensitivity");
  });

  it("saves an untouched prompt as no override at all", async () => {
    const onChange = await openEditor();
    await userEvent.click(screen.getByRole("button", { name: "Save prompt" }));
    expect(onChange).toHaveBeenCalledWith(undefined);
  });

  it("offers a reset that clears a stored override", async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <ClassifierPromptEditor systemPrompt="Grade data sensitivity" onChange={onChange} contextWindowSize={3} />,
    );
    expect(screen.getByRole("button", { name: "Edit custom prompt" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Reset to default" }));
    expect(onChange).toHaveBeenCalledWith(undefined);
  });

  it("seeds the editor from the stored override, not the default", async () => {
    await openEditor("Grade data sensitivity");
    expect(screen.getByLabelText("Classifier system prompt")).toHaveValue("Grade data sensitivity");
  });
});
