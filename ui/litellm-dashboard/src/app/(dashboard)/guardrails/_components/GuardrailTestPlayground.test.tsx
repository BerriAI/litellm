import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import GuardrailTestPlayground from "./GuardrailTestPlayground";

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

vi.mock("@/components/networking");

Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

describe("GuardrailTestPlayground", () => {
  const mockAccessToken = "test-token";
  const mockGuardrails = [
    {
      guardrail_id: "guard-1",
      guardrail_name: "test-guardrail",
      litellm_params: {
        guardrail: "presidio",
        mode: "pre_call",
        default_on: false,
      },
      guardrail_info: {},
    },
  ];

  beforeEach(() => {
    localization.language = "en";
    vi.clearAllMocks();
  });

  it("should allow selecting a guardrail and show test panel", async () => {
    /**
     * Tests that clicking on a guardrail selects it and displays the test panel.
     * This validates the core workflow of selecting and testing guardrails.
     */
    const user = userEvent.setup();

    render(
      <GuardrailTestPlayground
        guardrailsList={mockGuardrails}
        isLoading={false}
        accessToken={mockAccessToken}
        onClose={vi.fn()}
      />,
    );

    // Initially, the empty state should be shown
    expect(screen.getByText("Select Guardrails to Test")).toBeInTheDocument();

    // Click on the guardrail to select it
    const guardrailItem = screen.getByText("test-guardrail");
    await user.click(guardrailItem);

    // Verify the test panel is now shown
    await waitFor(() => {
      expect(screen.getByText("Test Guardrails:")).toBeInTheDocument();
      expect(screen.getByPlaceholderText("Enter text to test with guardrails...")).toBeInTheDocument();
    });

    // Verify the selected count
    expect(screen.getByText("1 of 1 selected")).toBeInTheDocument();
  });

  it("renders the testing playground in Russian", async () => {
    localization.language = "ru";
    const user = userEvent.setup();
    render(
      <GuardrailTestPlayground
        guardrailsList={mockGuardrails}
        isLoading={false}
        accessToken={mockAccessToken}
        onClose={vi.fn()}
      />,
    );

    expect(screen.getByText("Тестовая площадка ограничителей")).toBeInTheDocument();
    expect(screen.getByText("Выберите ограничители для проверки")).toBeInTheDocument();
    await user.click(screen.getByText("test-guardrail"));
    expect(await screen.findByPlaceholderText("Введите текст для проверки ограничителями...")).toBeInTheDocument();
    expect(screen.getByText("Выбрано: 1 из 1")).toBeInTheDocument();
  });
});
