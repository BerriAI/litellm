import { render, screen } from "@testing-library/react";
import type { i18n as I18n } from "i18next";
import { I18nextProvider, useTranslation } from "react-i18next";
import { afterEach, describe, expect, it, vi } from "vitest";

const POLYFILL = "@formatjs/intl-durationformat/polyfill-force.js";
const nativeDurationFormat = Object.getOwnPropertyDescriptor(Intl, "DurationFormat");

function Duration() {
  const { t } = useTranslation();
  return <span data-testid="duration">{t("format.duration", { val: 75_000 })}</span>;
}

const renderWithFreshI18n = async () => {
  vi.resetModules();
  const { default: i18n }: { default: I18n } = await import("./i18n");
  render(
    <I18nextProvider i18n={i18n}>
      <Duration />
    </I18nextProvider>,
  );
};

afterEach(() => {
  vi.doUnmock(POLYFILL);
  if (nativeDurationFormat) Object.defineProperty(Intl, "DurationFormat", nativeDurationFormat);
});

describe("Intl.DurationFormat polyfill", () => {
  it("re-renders durations once the polyfill arrives in a browser without Intl.DurationFormat", async () => {
    expect(nativeDurationFormat).toBeDefined();
    Reflect.deleteProperty(Intl, "DurationFormat");
    const { promise: arrival, resolve: arrive } = Promise.withResolvers<void>();
    vi.doMock(POLYFILL, async () => {
      await arrival;
      return vi.importActual(POLYFILL);
    });

    await renderWithFreshI18n();
    expect(screen.getByTestId("duration")).not.toHaveTextContent("1m 15s");

    arrive();
    expect(await screen.findByText("1m 15s")).toBeInTheDocument();
  });

  it("never loads the polyfill in a browser that has Intl.DurationFormat", async () => {
    const polyfill = vi.fn(() => vi.importActual(POLYFILL));
    vi.doMock(POLYFILL, polyfill);

    await renderWithFreshI18n();
    await vi.dynamicImportSettled();

    expect(screen.getByTestId("duration")).toHaveTextContent("1m 15s");
    expect(polyfill).not.toHaveBeenCalled();
  });
});
