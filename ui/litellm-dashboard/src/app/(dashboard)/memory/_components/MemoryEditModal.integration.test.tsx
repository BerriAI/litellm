import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { MemoryRow } from "@/components/networking";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import { translate, type Language } from "@/i18n/language";

import { MemoryEditModal } from "./MemoryEditModal";

const onSave = vi.fn<(key: string, value: string, metadataText: string, isCreate: boolean) => Promise<boolean>>();
const onClose = vi.fn();

const existingRow: MemoryRow = {
  memory_id: "mem-1",
  key: "user:profile",
  value: "The user prefers concise answers.",
  metadata: { tags: ["example"] },
};

const renderModal = (props: Partial<React.ComponentProps<typeof MemoryEditModal>> = {}) =>
  render(<MemoryEditModal open mode="create" onClose={onClose} onSave={onSave} {...props} />);

const fill = async (user: ReturnType<typeof userEvent.setup>, label: RegExp, text: string) => {
  await user.click(screen.getByLabelText(label));
  await user.paste(text);
};

describe("MemoryEditModal payload", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    onSave.mockResolvedValue(true);
  });

  it.each<Language>(["en", "zh-CN"])("已显示的校验随 %s 切换且保留记忆草稿", async (initialLanguage) => {
    const user = userEvent.setup();
    const nextLanguage = initialLanguage === "en" ? "zh-CN" : "en";
    const metadata = '{"Home":"Settings"}';
    render(
      <LanguageProvider defaultLanguage={initialLanguage} userId="memory-switch" storageScope="memory-switch">
        <LanguageSelector />
        <MemoryEditModal open mode="create" onClose={onClose} onSave={onSave} />
      </LanguageProvider>,
    );
    fireEvent.change(screen.getByLabelText(/^Metadata|^元数据/), { target: { value: metadata } });
    await user.click(screen.getByRole("button", { name: translate(initialLanguage, "Create"), exact: true }));
    expect(await screen.findByText(translate(initialLanguage, "Key is required"))).toBeVisible();
    expect(screen.getByText(translate(initialLanguage, "Value is required"))).toBeVisible();

    // 不重新提交或触发校验，已显示的本地错误也必须即时切换。
    fireEvent.change(screen.getByLabelText(translate(initialLanguage, "Interface language")), {
      target: { value: nextLanguage },
    });
    expect(screen.getByText(translate(nextLanguage, "Key is required"))).toBeVisible();
    expect(screen.getByText(translate(nextLanguage, "Value is required"))).toBeVisible();
    expect(screen.queryByText(translate(initialLanguage, "Key is required"))).not.toBeInTheDocument();
    expect(screen.getByLabelText(/^Metadata|^元数据/)).toHaveValue(metadata);
    expect(onSave).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText(/^Memory key|^键名/), { target: { value: "Home" } });
    fireEvent.change(screen.getByLabelText(/^Value|^值|^内容/), { target: { value: "Settings" } });
    await user.click(screen.getByRole("button", { name: translate(nextLanguage, "Create"), exact: true }));
    expect(onSave).toHaveBeenCalledWith("Home", "Settings", metadata, true);
  });

  it("中文校验后保存原始键名和内容", async () => {
    const user = userEvent.setup();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="memory-test" storageScope="memory-test">
        <MemoryEditModal open mode="create" onClose={onClose} onSave={onSave} />
      </LanguageProvider>,
    );
    await user.click(screen.getByRole("button", { name: "创建" }));
    expect(await screen.findByText("请输入键名")).toBeInTheDocument();
    expect(screen.getByText("请输入内容")).toBeInTheDocument();
    expect(onSave).not.toHaveBeenCalled();
    await fill(user, /^键名/, "Home");
    await fill(user, /^值|^内容/, "Home");
    await user.click(screen.getByRole("button", { name: "创建" }));
    expect(onSave).toHaveBeenCalledWith("Home", "Home", "", true);
  });

  it("sends the trimmed key, the value and the raw metadata text on create", async () => {
    const user = userEvent.setup();
    renderModal();

    await fill(user, /^Memory key/, "  user_role  ");
    await fill(user, /^Value/, "Remembers the user is an admin");
    await fill(user, /^Metadata/, '{"tags": ["example"]}');
    await user.click(screen.getByRole("button", { name: "Create" }));

    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onSave).toHaveBeenCalledWith("user_role", "Remembers the user is an admin", '{"tags": ["example"]}', true);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("sends an empty string for metadata the user never typed into", async () => {
    const user = userEvent.setup();
    renderModal();

    await fill(user, /^Memory key/, "user_role");
    await fill(user, /^Value/, "Remembers the user is an admin");
    await user.click(screen.getByRole("button", { name: "Create" }));

    expect(onSave).toHaveBeenCalledWith("user_role", "Remembers the user is an admin", "", true);
  });

  it("prefills from the row and sends the edited value with isCreate false", async () => {
    const user = userEvent.setup();
    renderModal({ mode: "edit", initialRow: existingRow });

    expect(await screen.findByLabelText(/^Memory key/)).toHaveValue("user:profile");
    expect(screen.getByLabelText(/^Memory key/)).toBeDisabled();
    expect(screen.getByLabelText(/^Value/)).toHaveValue("The user prefers concise answers.");
    expect(screen.getByLabelText(/^Metadata/)).toHaveValue('{\n  "tags": [\n    "example"\n  ]\n}');

    await user.clear(screen.getByLabelText(/^Value/));
    await fill(user, /^Value/, "The user prefers long answers.");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(onSave).toHaveBeenCalledWith(
      "user:profile",
      "The user prefers long answers.",
      '{\n  "tags": [\n    "example"\n  ]\n}',
      false,
    );
  });

  it("sends an empty metadata string for a row that has none", async () => {
    const user = userEvent.setup();
    renderModal({ mode: "edit", initialRow: { ...existingRow, metadata: null } });

    expect(await screen.findByLabelText(/^Metadata/)).toHaveValue("");

    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(onSave).toHaveBeenCalledWith("user:profile", "The user prefers concise answers.", "", false);
  });

  it("keeps the modal open when the save is rejected by the caller", async () => {
    const user = userEvent.setup();
    onSave.mockResolvedValue(false);
    renderModal();

    await fill(user, /^Memory key/, "user_role");
    await fill(user, /^Value/, "Remembers the user is an admin");
    await user.click(screen.getByRole("button", { name: "Create" }));

    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onClose).not.toHaveBeenCalled();
  });

  it("reports each required field as soon as it is emptied", async () => {
    const user = userEvent.setup();
    renderModal();

    await fill(user, /^Memory key/, "user_role");
    await fill(user, /^Value/, "Remembers the user is an admin");
    expect(screen.queryByText("Key is required")).not.toBeInTheDocument();

    await user.clear(screen.getByLabelText(/^Memory key/));
    expect(await screen.findByText("Key is required")).toBeInTheDocument();

    await user.clear(screen.getByLabelText(/^Value/));
    expect(await screen.findByText("Value is required")).toBeInTheDocument();
    expect(onSave).not.toHaveBeenCalled();
  });
});
