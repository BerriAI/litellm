import React from "react";
import { fireEvent, screen } from "@testing-library/react";
import { renderWithProviders } from "@/../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AddGuardrailForm from "./add_guardrail_form";

const localization = vi.hoisted(() => ({ language: "en" as "en" | "ru" }));

vi.mock("react-i18next", async () => {
  const { resources } = await import("@/i18n/catalog");
  const t = (key: string, values?: Record<string, unknown>) => {
    const copy = key.split(".").reduce<unknown>((value, segment) => {
      if (typeof value !== "object" || value === null) return undefined;
      return (value as Record<string, unknown>)[segment];
    }, resources[localization.language].gateway);
    if (typeof copy !== "string") return key;
    return Object.entries(values ?? {}).reduce(
      (text, [name, value]) => text.replaceAll(`{{${name}}}`, String(value)),
      copy,
    );
  };
  return { useTranslation: () => ({ t }) };
});

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
    localization.language = "en";
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
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("AddGuardrailForm provider options", () => {
  beforeEach(() => {
    localization.language = "en";
    vi.clearAllMocks();
  });

  it("renders provider options with logos from the bundled guardrail logo map", async () => {
    renderForm();
    fireEvent.mouseDown(screen.getByLabelText("Guardrail Provider"));

    const logo = await screen.findByAltText("Presidio PII logo");
    expect(logo.getAttribute("src")).toContain("microsoft_azure.svg");
  });

  it("renders the creation wizard in Russian", () => {
    localization.language = "ru";
    renderForm();

    expect(screen.getByText("Создать ограничитель")).toBeInTheDocument();
    expect(screen.getByText("Основные сведения")).toBeInTheDocument();
    expect(screen.getByLabelText("Название ограничителя")).toBeInTheDocument();
    expect(screen.getByLabelText("Провайдер ограничителя")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Далее" })).toBeInTheDocument();
  });
});
