import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CustomCodeModal from "./CustomCodeModal";
import { createGuardrailCall, updateGuardrailCall, testCustomCodeGuardrail } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  createGuardrailCall: vi.fn(),
  updateGuardrailCall: vi.fn(),
  testCustomCodeGuardrail: vi.fn(),
}));

const mockCreate = vi.mocked(createGuardrailCall);
const mockUpdate = vi.mocked(updateGuardrailCall);
const mockTest = vi.mocked(testCustomCodeGuardrail);

describe("CustomCodeModal", () => {
  const onClose = vi.fn();
  const onSuccess = vi.fn();

  const renderModal = (overrides = {}) =>
    render(<CustomCodeModal visible onClose={onClose} onSuccess={onSuccess} accessToken="test-token" {...overrides} />);

  beforeEach(() => {
    vi.clearAllMocks();
    mockCreate.mockResolvedValue({} as never);
    mockUpdate.mockResolvedValue({} as never);
  });

  it("should render the create heading and the editor scaffolding", async () => {
    renderModal();

    expect(await screen.findByText("Create Custom Guardrail")).toBeInTheDocument();
    expect(screen.getByText("Define custom logic using Python-like syntax")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("e.g., block-pii-custom")).toBeInTheDocument();
    expect(screen.getByText("Guardrail Name")).toBeInTheDocument();
    expect(screen.getByText("Mode (can select multiple)")).toBeInTheDocument();
    expect(screen.getByText("Available Primitives")).toBeInTheDocument();
    expect(screen.getByText("Python Logic")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /save guardrail/i })).toBeInTheDocument();
  });

  it("should seed the editor with the empty template", async () => {
    renderModal();

    const editor = await screen.findByDisplayValue(/async def apply_guardrail/);
    expect(editor).toBeInTheDocument();
  });

  it("should not render its content when not visible", () => {
    renderModal({ visible: false });

    expect(screen.queryByText("Create Custom Guardrail")).not.toBeInTheDocument();
  });

  it("should render the edit heading and existing values in edit mode", async () => {
    renderModal({
      editData: {
        guardrail_id: "g-1",
        guardrail_name: "existing-guardrail",
        litellm_params: { mode: "post_call", default_on: true, custom_code: "def apply_guardrail(): pass" },
      },
    });

    expect(await screen.findByText("Edit Custom Guardrail")).toBeInTheDocument();
    expect(screen.getByDisplayValue("existing-guardrail")).toBeInTheDocument();
    expect(screen.getByDisplayValue("def apply_guardrail(): pass")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /update guardrail/i })).toBeInTheDocument();
  });

  it("should open without crashing for a tag-scoped mode and show it read-only", async () => {
    renderModal({
      editData: {
        guardrail_id: "g-tag",
        guardrail_name: "tag-mode-guardrail",
        litellm_params: {
          mode: { tags: { "team-a": "pre_call" }, default: "post_call" },
          default_on: true,
          custom_code: "def apply_guardrail(): pass",
        },
      },
    });

    expect(await screen.findByText("Edit Custom Guardrail")).toBeInTheDocument();
    const modeInput = screen.getByLabelText("Mode (tag-scoped)");
    expect(modeInput).toBeDisabled();
    expect(modeInput).toHaveValue("post_call, pre_call (tag-based)");
    expect(screen.getByText("Mode (tag-scoped, read-only)")).toBeInTheDocument();
    expect(screen.queryByText(/applies to/)).not.toBeInTheDocument();
  });

  it("should omit mode and stream_scope from the update payload for a tag-scoped guardrail", async () => {
    const user = userEvent.setup();
    renderModal({
      editData: {
        guardrail_id: "g-tag",
        guardrail_name: "tag-mode-guardrail",
        litellm_params: {
          mode: { tags: { "team-a": "pre_call" }, default: "post_call" },
          default_on: true,
          custom_code: "def apply_guardrail(): pass",
        },
      },
    });

    const nameInput = await screen.findByDisplayValue("tag-mode-guardrail");
    await user.clear(nameInput);
    await user.type(nameInput, "renamed-guardrail");
    await user.click(screen.getByRole("button", { name: /update guardrail/i }));

    await waitFor(() => {
      expect(mockUpdate).toHaveBeenCalledTimes(1);
    });
    const [token, guardrailId, payload] = mockUpdate.mock.calls[0] as [
      string,
      string,
      Record<string, Record<string, unknown>>,
    ];
    expect(token).toBe("test-token");
    expect(guardrailId).toBe("g-tag");
    expect(payload.guardrail_name).toBe("renamed-guardrail");
    expect(payload.litellm_params.custom_code).toBe("def apply_guardrail(): pass");
    expect(payload.litellm_params).not.toHaveProperty("mode");
    expect(payload.litellm_params).not.toHaveProperty("stream_scope");
  });

  it("should keep save disabled until a guardrail name is entered", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole("button", { name: /save guardrail/i }));
    expect(mockCreate).not.toHaveBeenCalled();

    await user.type(screen.getByPlaceholderText("e.g., block-pii-custom"), "my-guardrail");
    await user.click(screen.getByRole("button", { name: /save guardrail/i }));

    await waitFor(() => {
      expect(mockCreate).toHaveBeenCalledTimes(1);
    });
  });

  it("should create the guardrail with the entered name, mode and code", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.type(await screen.findByPlaceholderText("e.g., block-pii-custom"), "block-pii");
    await user.click(screen.getByRole("button", { name: /save guardrail/i }));

    await waitFor(() => {
      expect(mockCreate).toHaveBeenCalled();
    });

    const [token, payload] = mockCreate.mock.calls[0] as [string, Record<string, never>];
    expect(token).toBe("test-token");
    expect(payload).toMatchObject({
      guardrail_name: "block-pii",
      litellm_params: { guardrail: "custom_code", mode: ["pre_call"], default_on: false },
    });
    await waitFor(() => {
      expect(onSuccess).toHaveBeenCalled();
    });
  });

  it("should switch the editor contents when a template is chosen", async () => {
    const user = userEvent.setup();
    renderModal();

    expect(await screen.findByDisplayValue(/async def apply_guardrail/)).toBeInTheDocument();

    await user.click(screen.getByRole("combobox", { name: "Template" }));
    const options = await screen.findAllByText("Block SSN");
    await user.click(options[options.length - 1]);

    expect(await screen.findByDisplayValue(/SSN detected/)).toBeInTheDocument();
  });

  it("should narrow the mode options to the typed search text", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(screen.getAllByRole("combobox")[0]);
    await user.keyboard("mcp");

    expect(await screen.findByText("pre_mcp_call (Before MCP Tool Call)")).toBeInTheDocument();
    expect(screen.queryByText("logging_only")).not.toBeInTheDocument();
  });

  it("should update the logging-only scope in edit mode", async () => {
    const user = userEvent.setup();
    renderModal({
      editData: {
        guardrail_id: "g-1",
        guardrail_name: "existing-guardrail",
        litellm_params: {
          mode: ["logging_only"],
          logging_only_scope: "input",
          custom_code: "def apply_guardrail(): pass",
        },
      },
    });

    const scopeSelect = await screen.findByRole("combobox", { name: "Logging only scope" });
    expect(scopeSelect).toHaveTextContent("Input only (request)");

    await user.click(scopeSelect);
    await user.click(await screen.findByRole("option", { name: "Output only (response)" }));
    await user.click(screen.getByRole("button", { name: /update guardrail/i }));

    await waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(mockUpdate.mock.calls[0][2]).toMatchObject({
      litellm_params: { logging_only_scope: "output" },
    });
  });

  it("should clear the logging-only scope when Default is selected in edit mode", async () => {
    const user = userEvent.setup();
    renderModal({
      editData: {
        guardrail_id: "g-1",
        guardrail_name: "existing-guardrail",
        litellm_params: {
          mode: ["logging_only"],
          logging_only_scope: "input",
          custom_code: "def apply_guardrail(): pass",
        },
      },
    });

    await user.click(await screen.findByRole("combobox", { name: "Logging only scope" }));
    await user.click(await screen.findByRole("option", { name: "Default (request and response)" }));
    await user.click(screen.getByRole("button", { name: /update guardrail/i }));

    await waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(mockUpdate.mock.calls[0][2]).toHaveProperty("litellm_params.logging_only_scope", null);
  });

  it("should omit an unchanged logging-only scope from the edit payload", async () => {
    renderModal({
      editData: {
        guardrail_id: "g-1",
        guardrail_name: "existing-guardrail",
        litellm_params: {
          mode: ["logging_only"],
          logging_only_scope: "input",
          custom_code: "def apply_guardrail(): pass",
        },
      },
    });

    fireEvent.change(await screen.findByPlaceholderText("e.g., block-pii-custom"), {
      target: { value: "renamed-guardrail" },
    });
    await userEvent.setup().click(screen.getByRole("button", { name: /update guardrail/i }));

    await waitFor(() => expect(mockUpdate).toHaveBeenCalledTimes(1));
    expect(mockUpdate.mock.calls[0][2]).not.toHaveProperty("litellm_params.logging_only_scope");
  });

  it("should not show logging-only scope for other modes in edit mode", async () => {
    renderModal({
      editData: {
        guardrail_id: "g-1",
        guardrail_name: "existing-guardrail",
        litellm_params: { mode: "pre_call", custom_code: "def apply_guardrail(): pass" },
      },
    });

    expect(await screen.findByText("Edit Custom Guardrail")).toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "Logging only scope" })).not.toBeInTheDocument();
  });

  it("should create a logging-only guardrail with its selected scope", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(screen.getAllByRole("combobox")[0]);
    await user.keyboard("logging_only");
    await user.click(await screen.findByRole("option", { name: "logging_only" }));
    await user.click(await screen.findByRole("combobox", { name: "Logging only scope" }));
    await user.click(await screen.findByRole("option", { name: "Input only (request)" }));
    fireEvent.change(screen.getByPlaceholderText("e.g., block-pii-custom"), {
      target: { value: "logging-only-guardrail" },
    });
    await user.click(screen.getByRole("button", { name: /save guardrail/i }));

    await waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    expect(mockCreate.mock.calls[0][1]).toMatchObject({
      litellm_params: { logging_only_scope: "input" },
    });
  });

  it("should omit the default logging-only scope when creating a guardrail", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(screen.getAllByRole("combobox")[0]);
    await user.keyboard("logging_only");
    await user.click(await screen.findByRole("option", { name: "logging_only" }));
    await screen.findByRole("combobox", { name: "Logging only scope" });
    fireEvent.change(screen.getByPlaceholderText("e.g., block-pii-custom"), {
      target: { value: "logging-only-guardrail" },
    });
    await user.click(screen.getByRole("button", { name: /save guardrail/i }));

    await waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    expect(mockCreate.mock.calls[0][1]).not.toHaveProperty("litellm_params.logging_only_scope");
  });

  it("should expand the test section and run a test against the backend", async () => {
    const user = userEvent.setup();
    mockTest.mockResolvedValue({ success: true, result: { action: "allow" } } as never);
    renderModal();

    await user.click(await screen.findByText("Test Your Guardrail"));

    const runButton = await screen.findByRole("button", { name: /run test/i });
    await user.click(runButton);

    await waitFor(() => {
      expect(mockTest).toHaveBeenCalled();
    });
    expect(await screen.findByText("Allowed")).toBeInTheDocument();
  });

  it("should surface a backend test error", async () => {
    const user = userEvent.setup();
    mockTest.mockResolvedValue({ success: false, error: "boom", error_type: "SyntaxError" } as never);
    renderModal();

    await user.click(await screen.findByText("Test Your Guardrail"));
    await user.click(await screen.findByRole("button", { name: /run test/i }));

    expect(await screen.findByText("boom")).toBeInTheDocument();
    expect(screen.getByText("[SyntaxError]")).toBeInTheDocument();
  });

  it("should cancel through the footer button", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.click(await screen.findByRole("button", { name: "Cancel" }));

    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
