import { fireEvent, render, screen } from "@testing-library/react";
import i18next from "i18next";
import { I18nextProvider } from "react-i18next";
import { describe, expect, it, vi } from "vitest";
import en from "@/i18n/locales/en.json";
import zh from "@/i18n/locales/zh-CN.json";
import PromptMessagesCard from "./PromptMessagesCard";

vi.mock("../variable_textarea", () => ({
  default: (props: any) => <textarea value={props.value} onChange={(event) => props.onChange(event.target.value)} />,
}));

describe("PromptMessagesCard", () => {
  it("renders messages and adds another message", () => {
    const onAddMessage = vi.fn();
    render(
      <PromptMessagesCard
        messages={[{ role: "user", content: "Hello" }]}
        onAddMessage={onAddMessage}
        onUpdateMessage={vi.fn()}
        onRemoveMessage={vi.fn()}
        onMoveMessage={vi.fn()}
      />,
    );
    expect(screen.getByDisplayValue("Hello")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /add message/i }));
    expect(onAddMessage).toHaveBeenCalledOnce();
  });

  it.each([
    ["user", "User"],
    ["assistant", "Assistant"],
    ["system", "System"],
  ])("keeps the %s role name in English on a Chinese dashboard", async (role, label) => {
    const i18n = i18next.createInstance();
    await i18n.init({ lng: "zh-CN", resources: { en: { translation: en }, "zh-CN": { translation: zh } } });
    render(
      <I18nextProvider i18n={i18n}>
        <PromptMessagesCard
          messages={[{ role, content: "Hello" }]}
          onAddMessage={vi.fn()}
          onUpdateMessage={vi.fn()}
          onRemoveMessage={vi.fn()}
          onMoveMessage={vi.fn()}
        />
      </I18nextProvider>,
    );

    expect(screen.getByRole("combobox", { name: "消息 1 的角色" })).toHaveTextContent(label);
  });
});
