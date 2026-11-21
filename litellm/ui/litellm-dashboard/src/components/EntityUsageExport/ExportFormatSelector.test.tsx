import i18next from "i18next";
import { I18nextProvider } from "react-i18next";

import en from "@/i18n/locales/en.json";
import zh from "@/i18n/locales/zh-CN.json";

import { renderWithProviders, screen } from "../../../tests/test-utils";
import { vi } from "vitest";
import ExportFormatSelector from "./ExportFormatSelector";

describe("ExportFormatSelector", () => {
  it("should render", () => {
    renderWithProviders(<ExportFormatSelector value="csv" onChange={vi.fn()} />);
    expect(screen.getByText("Format")).toBeInTheDocument();
  });

  it("should display the current value", () => {
    renderWithProviders(<ExportFormatSelector value="csv" onChange={vi.fn()} />);
    expect(screen.getByText("CSV (Excel, Google Sheets)")).toBeInTheDocument();
  });

  it("should display JSON label when json is selected", () => {
    renderWithProviders(<ExportFormatSelector value="json" onChange={vi.fn()} />);
    expect(screen.getByText("JSON (includes metadata)")).toBeInTheDocument();
  });

  it("keeps spreadsheet product names in English in the Chinese export form", async () => {
    const i18n = i18next.createInstance();
    await i18n.init({
      lng: "zh-CN",
      fallbackLng: "en",
      resources: { en: { translation: en }, "zh-CN": { translation: zh } },
    });

    renderWithProviders(
      <I18nextProvider i18n={i18n}>
        <ExportFormatSelector value="csv" onChange={vi.fn()} />
      </I18nextProvider>,
    );

    expect(screen.getByText("格式")).toBeInTheDocument();
    expect(screen.getByText("CSV (Excel, Google Sheets)")).toBeInTheDocument();
  });
});
