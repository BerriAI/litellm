import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import BudgetDurationDropdown from "./budget_duration_dropdown";
import { LanguageProvider } from "@/contexts/LanguageContext";

describe("BudgetDurationDropdown", () => {
  it.each([undefined, "Home"])("汉化默认占位但保留调用方提供的文案 %s", (placeholder) => {
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="duration-i18n" storageScope="duration-i18n">
        <BudgetDurationDropdown value={null} placeholder={placeholder} />
      </LanguageProvider>,
    );
    expect(screen.getByRole("combobox")).toHaveTextContent(placeholder ?? "不重置");
  });

  it("labels an unset period as Never resets in the trigger and option", async () => {
    const user = userEvent.setup();
    render(<BudgetDurationDropdown value={null} />);

    expect(screen.getByRole("combobox")).toHaveTextContent("Never resets");

    await user.click(screen.getByRole("combobox"));

    expect(await screen.findByRole("option", { name: "Never resets" })).toBeInTheDocument();
  });

  it("emits null when Never resets is selected", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<BudgetDurationDropdown value="7d" onChange={onChange} />);

    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByRole("option", { name: "Never resets" }));

    expect(onChange).toHaveBeenCalledOnce();
    expect(onChange.mock.calls[0][0]).toBeNull();
  });

  it("uses a supplied placeholder for the unset trigger and option", async () => {
    const user = userEvent.setup();
    render(<BudgetDurationDropdown value={null} placeholder="Not set" />);

    expect(screen.getByRole("combobox")).toHaveTextContent("Not set");

    await user.click(screen.getByRole("combobox"));

    expect(await screen.findByRole("option", { name: "Not set" })).toBeInTheDocument();
  });
});
