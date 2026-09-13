import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { i18n } from "i18next";

// Control i18n initialisation resolves so we can assert the readiness gate
// blocks the business subtree before the instance is ready.
let releaseGate: (instance: i18n) => void = () => {};
let pending: Promise<i18n> | null = null;

vi.mock("./i18n", () => {
  return {
    getI18n: () =>
      (pending ??= new Promise<i18n>((resolve) => {
        releaseGate = resolve;
      })),
  };
});

import { I18nProvider, useI18n } from "@/i18n";

function GateConsumer() {
  const { locale } = useI18n();
  return <span data-testid="gate-child">{locale}</span>;
}

const ORIGINAL_LANG = "en";

beforeEach(() => {
  localStorage.clear();
  document.documentElement.lang = ORIGINAL_LANG;
  pending = null;
});

afterEach(() => {
  vi.resetModules();
  pending = null;
  localStorage.clear();
  document.documentElement.lang = ORIGINAL_LANG;
});

describe("I18nProvider readiness gate", () => {
  it("does not render the business subtree before the instance is ready", () => {
    localStorage.setItem("litellm.locale", "zh-CN");
    render(
      <I18nProvider>
        <GateConsumer />
      </I18nProvider>,
    );

    // While i18n initialisation is pending, children must not appear (no key exposure).
    expect(screen.queryByTestId("gate-child")).not.toBeInTheDocument();
  });

  it("renders children once initialisation resolves", async () => {
    localStorage.setItem("litellm.locale", "zh-CN");
    render(
      <I18nProvider>
        <GateConsumer />
      </I18nProvider>,
    );

    expect(screen.queryByTestId("gate-child")).not.toBeInTheDocument();

    // Resolve the pending initialisation with a minimal fake instance.
    releaseGate({
      isInitialized: true,
      language: "zh-CN",
      changeLanguage: vi.fn(async () => {}),
      t: vi.fn(),
    } as unknown as i18n);

    await waitFor(() => expect(screen.getByTestId("gate-child")).toBeInTheDocument());
  });
});
