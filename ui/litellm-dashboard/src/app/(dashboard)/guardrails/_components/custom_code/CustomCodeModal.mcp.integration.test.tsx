import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CustomCodeModal from "./CustomCodeModal";
import { createGuardrailCall, updateGuardrailCall } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  createGuardrailCall: vi.fn(),
  updateGuardrailCall: vi.fn(),
  testCustomCodeGuardrail: vi.fn(),
}));

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(createGuardrailCall).mockResolvedValue({} as never);
  vi.mocked(updateGuardrailCall).mockResolvedValue({} as never);
});

describe("MCP block policy in the custom-code editor", () => {
  it("sends Stop under litellm_params when creating an MCP guardrail", async () => {
    const user = userEvent.setup({ delay: null });
    render(<CustomCodeModal visible onClose={vi.fn()} onSuccess={vi.fn()} accessToken="test-token" />);
    fireEvent.change(screen.getByPlaceholderText("e.g., block-pii-custom"), { target: { value: "tool-policy" } });
    await user.click(screen.getByRole("combobox", { name: "" }));
    await user.click(await screen.findByRole("option", { name: /pre_mcp_call/ }));
    await user.keyboard("{Escape}");
    await user.click(await screen.findByLabelText("When an MCP tool is blocked"));
    await user.click(await screen.findByRole("option", { name: "Stop the request with an error" }));
    await user.click(screen.getByRole("button", { name: /save guardrail/i }));
    await waitFor(() =>
      expect(createGuardrailCall).toHaveBeenCalledWith(
        "test-token",
        expect.objectContaining({
          litellm_params: expect.objectContaining({
            on_mcp_block: "raise",
            mode: expect.arrayContaining(["pre_mcp_call"]),
          }),
        }),
      ),
    );
  });

  it("shows a tag-scoped saved Stop setting and sends an explicit reset on edit", async () => {
    const user = userEvent.setup({ delay: null });
    render(
      <CustomCodeModal
        visible
        onClose={vi.fn()}
        onSuccess={vi.fn()}
        accessToken="test-token"
        editData={{
          guardrail_id: "tool-policy-id",
          guardrail_name: "tool-policy",
          litellm_params: {
            mode: { tags: { tools: "during_mcp_call" }, default: "pre_call" },
            on_mcp_block: "raise",
            custom_code: "def apply_guardrail(inputs, request_data, input_type): return allow()",
          },
        }}
      />,
    );
    const control = await screen.findByLabelText("When an MCP tool is blocked");
    expect(control).toHaveTextContent("Stop the request with an error");
    await user.click(control);
    await user.click(await screen.findByRole("option", { name: "Return a tool error and let the model continue" }));
    await user.click(screen.getByRole("button", { name: /update guardrail/i }));
    await waitFor(() =>
      expect(updateGuardrailCall).toHaveBeenCalledWith(
        "test-token",
        "tool-policy-id",
        expect.objectContaining({
          litellm_params: expect.objectContaining({ on_mcp_block: "tool_error" }),
        }),
      ),
    );
  });
});
