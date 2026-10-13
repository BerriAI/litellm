import { beforeEach, describe, expect, it, vi } from "vitest";
import { GuardrailDeleteModal } from "./GuardrailDeleteModal";
import { deleteGuardrailCall } from "@/components/networking";
import { toast } from "@/lib/toast";
import { fireEvent, renderWithProviders, screen, waitFor } from "@/../tests/test-utils";

vi.mock("@/components/networking", () => ({ deleteGuardrailCall: vi.fn() }));
vi.mock("@/lib/toast", () => ({ toast: { success: vi.fn(), fromError: vi.fn() } }));

const GUARDRAIL = {
  guardrail_id: "gr-1",
  guardrail_name: "word-block",
  litellm_params: { guardrail: "litellm_content_filter", mode: "pre_call", default_on: true },
};

const renderModal = () => {
  const onClose = vi.fn();
  const onDeleted = vi.fn();
  renderWithProviders(
    <GuardrailDeleteModal guardrail={GUARDRAIL} accessToken="tok" onClose={onClose} onDeleted={onDeleted} />,
  );
  return { onClose, onDeleted };
};

const confirmDelete = () => {
  fireEvent.change(screen.getByPlaceholderText(GUARDRAIL.guardrail_name), {
    target: { value: GUARDRAIL.guardrail_name },
  });
  fireEvent.click(screen.getByRole("button", { name: "Delete" }));
};

describe("GuardrailDeleteModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("warns about name references and only enables Delete once the exact name is typed", () => {
    renderModal();

    expect(screen.getByText(/keys, teams, or policies that reference/i)).toBeInTheDocument();
    const confirmButton = screen.getByRole("button", { name: "Delete" });
    expect(confirmButton).toBeDisabled();
    fireEvent.change(screen.getByPlaceholderText(GUARDRAIL.guardrail_name), { target: { value: "word" } });
    expect(confirmButton).toBeDisabled();
    fireEvent.change(screen.getByPlaceholderText(GUARDRAIL.guardrail_name), {
      target: { value: GUARDRAIL.guardrail_name },
    });
    expect(confirmButton).toBeEnabled();
  });

  it("shows a success toast naming the guardrail and calls onDeleted after a successful delete", async () => {
    vi.mocked(deleteGuardrailCall).mockResolvedValue({});
    const { onClose, onDeleted } = renderModal();

    confirmDelete();

    await waitFor(() => expect(onDeleted).toHaveBeenCalledTimes(1));
    expect(deleteGuardrailCall).toHaveBeenCalledWith("tok", GUARDRAIL.guardrail_id);
    expect(toast.success).toHaveBeenCalledWith('Guardrail "word-block" deleted successfully');
    expect(toast.fromError).not.toHaveBeenCalled();
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("shows an error toast and does not call onDeleted when the delete fails", async () => {
    vi.mocked(deleteGuardrailCall).mockRejectedValue(new Error("boom"));
    const { onClose, onDeleted } = renderModal();

    confirmDelete();

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(toast.fromError).toHaveBeenCalledWith("Failed to delete guardrail");
    expect(toast.success).not.toHaveBeenCalled();
    expect(onDeleted).not.toHaveBeenCalled();
  });
});
