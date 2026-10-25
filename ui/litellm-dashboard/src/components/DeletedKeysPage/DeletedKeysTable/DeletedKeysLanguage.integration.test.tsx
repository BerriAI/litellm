import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../tests/test-utils";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import type { DeletedKeyResponse } from "@/app/(dashboard)/hooks/keys/useKeys";
import { DeletedKeysTable } from "./DeletedKeysTable";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

function LanguageSwitch() {
  const { setPreference } = useLanguage();
  return <button onClick={() => setPreference("en")}>English</button>;
}

beforeEach(() => localStorage.clear());

it("切换语言只改变固定表头，不改变密钥别名、链接或分页参数", async () => {
  const user = userEvent.setup();
  const onPaginationChange = vi.fn();
  // 仅填充本次表格交互实际消费的响应字段。
  const key = {
    token: "key-opaque-id",
    key_alias: "Home",
    user_id: "account-1",
    spend: 0,
    max_budget: null,
    deleted_at: "2026-01-01T00:00:00Z",
  } as DeletedKeyResponse;
  renderWithProviders(
    <LanguageProvider defaultLanguage="zh-CN" userId="language-test" storageScope="logs-language-test">
      <LanguageSwitch />
      <DeletedKeysTable
        keys={[key]}
        totalCount={120}
        isLoading={false}
        pagination={{ pageIndex: 0, pageSize: 50 }}
        onPaginationChange={onPaginationChange}
      />
    </LanguageProvider>,
  );

  expect(screen.getByRole("columnheader", { name: "密钥别名" })).toBeInTheDocument();
  expect(screen.getByText("Home")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "account-1" })).toHaveAttribute("href", "/ui/users?user=account-1");
  await user.click(screen.getByTestId("pagination-next"));
  expect(onPaginationChange).toHaveBeenCalled();
  const update = onPaginationChange.mock.calls[0][0];
  expect(typeof update === "function" ? update({ pageIndex: 0, pageSize: 50 }) : update).toEqual({
    pageIndex: 1,
    pageSize: 50,
  });

  await user.click(screen.getByRole("button", { name: "English" }));
  expect(screen.getByRole("columnheader", { name: "Key Alias" })).toBeInTheDocument();
  expect(screen.getByText("Home")).toBeInTheDocument();
});
