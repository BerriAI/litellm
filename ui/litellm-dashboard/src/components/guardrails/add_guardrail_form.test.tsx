import React from "react";
import { fireEvent, screen } from "@testing-library/react";
import { renderWithProviders } from "../../../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AddGuardrailForm from "./add_guardrail_form";

vi.mock("@/components/networking", () => ({
  createGuardrailCall: vi.fn(),
  getGuardrailProviderSpecificParams: vi.fn().mockResolvedValue({}),
  getGuardrailUISettings: vi.fn().mockResolvedValue({}),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
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

    const wrap = document.querySelector(".ant-modal-wrap") as HTMLElement;
    expect(wrap).toBeTruthy();
    fireEvent.mouseDown(wrap);
    fireEvent.mouseUp(wrap);
    fireEvent.click(wrap);

    expect(onClose).not.toHaveBeenCalled();
  });

  it("closes when the user clicks the explicit close button", () => {
    const { onClose } = renderForm();
    fireEvent.click(screen.getByRole("button", { name: "✕" }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
