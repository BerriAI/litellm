import { fireEvent, renderWithProviders, screen } from "../../../tests/test-utils";
import { vi } from "vitest";
import CustomTierPromptEditor from "./CustomTierPromptEditor";

const { getAutoRouterCustomTierPromptCall } = vi.hoisted(() => ({
  getAutoRouterCustomTierPromptCall: vi.fn(),
}));

vi.mock("@/components/networking", () => ({ getAutoRouterCustomTierPromptCall }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test" }),
}));

const tierRows = [
  { id: "SIMPLE", name: "SIMPLE", definition: "", models: ["haiku"] },
  { id: "audit", name: "AUDIT", definition: "security review", models: ["opus"] },
];

const renderEditor = (classificationPrompt?: string) => {
  const onChange = vi.fn();
  renderWithProviders(
    <CustomTierPromptEditor
      classificationPrompt={classificationPrompt}
      onChange={onChange}
      tierRows={tierRows}
      contextWindowSize={3}
    />,
  );
  return onChange;
};

beforeEach(() => {
  vi.clearAllMocks();
  getAutoRouterCustomTierPromptCall.mockResolvedValue(
    "Route for payments.\n\nTiers:\n- SIMPLE: greetings, chitchat\n- AUDIT: security review",
  );
});

describe("CustomTierPromptEditor", () => {
  it("shows the prompt the proxy assembled rather than one rebuilt in the browser", async () => {
    renderEditor();
    fireEvent.click(screen.getByRole("button", { name: "Edit prompt" }));

    // The blank SIMPLE row inherits criteria that live only in the backend, so a preview built here
    // could not show them. Asserting the rendered text comes from the response is what pins that.
    expect(await screen.findByLabelText("Assembled classifier prompt")).toHaveTextContent(
      "- SIMPLE: greetings, chitchat",
    );
  });

  it("sends a blank built-in definition as an absent description, which is what inherits the criteria", async () => {
    renderEditor();
    fireEvent.click(screen.getByRole("button", { name: "Edit prompt" }));
    await screen.findByLabelText("Assembled classifier prompt");

    expect(getAutoRouterCustomTierPromptCall).toHaveBeenCalledWith(
      "sk-test",
      3,
      [{ name: "SIMPLE" }, { name: "AUDIT", description: "security review" }],
      "",
    );
  });

  it("previews the draft being typed, not only the saved prompt", async () => {
    renderEditor("saved opening");
    fireEvent.click(screen.getByRole("button", { name: "Edit prompt" }));
    await screen.findByLabelText("Assembled classifier prompt");

    fireEvent.change(screen.getByLabelText("Classifier opening instructions"), { target: { value: "edited opening" } });

    await vi.waitFor(() =>
      expect(getAutoRouterCustomTierPromptCall).toHaveBeenLastCalledWith(
        "sk-test",
        3,
        expect.anything(),
        "edited opening",
      ),
    );
  });

  it("keeps the editor usable when the preview cannot be fetched", async () => {
    getAutoRouterCustomTierPromptCall.mockRejectedValue(new Error("boom"));
    renderEditor();
    fireEvent.click(screen.getByRole("button", { name: "Edit prompt" }));

    expect(await screen.findByRole("button", { name: "Save prompt" })).toBeEnabled();
    expect(screen.queryByLabelText("Assembled classifier prompt")).not.toBeInTheDocument();
  });

  it("saves the draft as the router's opening instructions", async () => {
    const onChange = renderEditor();
    fireEvent.click(screen.getByRole("button", { name: "Edit prompt" }));
    fireEvent.change(screen.getByLabelText("Classifier opening instructions"), { target: { value: "  my rubric  " } });
    fireEvent.click(screen.getByRole("button", { name: "Save prompt" }));

    expect(onChange).toHaveBeenCalledWith("my rubric");
  });

  it("clears the prompt rather than saving whitespace, so the router keeps the built-in opening", () => {
    const onChange = renderEditor("saved opening");
    fireEvent.click(screen.getByRole("button", { name: "Reset to default" }));

    expect(onChange).toHaveBeenCalledWith(undefined);
  });
});
