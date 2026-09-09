import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
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

    const comboboxes = screen.getAllByRole("combobox");
    await user.click(comboboxes[comboboxes.length - 1]);
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
