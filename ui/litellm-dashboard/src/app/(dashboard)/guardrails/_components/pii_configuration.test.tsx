import { render } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import PiiConfiguration from "./pii_configuration";

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

describe("PiiConfiguration", () => {
  it("should render", () => {
    localization.language = "en";
    const { getByText } = render(
      <PiiConfiguration
        entities={[]}
        actions={[]}
        selectedEntities={[]}
        selectedActions={{}}
        onEntitySelect={() => {}}
        onActionSelect={() => {}}
        entityCategories={[]}
      />,
    );
    expect(getByText("Configure PII Protection")).toBeInTheDocument();
  });

  it("renders the PII configuration in Russian", () => {
    localization.language = "ru";
    const { getByText } = render(
      <PiiConfiguration
        entities={[]}
        actions={[]}
        selectedEntities={[]}
        selectedActions={{}}
        onEntitySelect={() => {}}
        onActionSelect={() => {}}
        entityCategories={[]}
      />,
    );
    expect(getByText("Настройка защиты персональных данных")).toBeInTheDocument();
    expect(getByText("Быстрые действия")).toBeInTheDocument();
  });
});
