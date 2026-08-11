/* @vitest-environment jsdom */
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { UsersTable } from "./UsersTable";

vi.mock("react-i18next", async () => {
  const { resources } = await import("@/i18n/catalog");
  const t = (key: string, values?: Record<string, unknown>) => {
    const copy = key.split(".").reduce<unknown>((value, segment) => {
      if (typeof value !== "object" || value === null) return undefined;
      return (value as Record<string, unknown>)[segment];
    }, resources.ru.gateway);

    if (typeof copy !== "string") return key;
    return Object.entries(values ?? {}).reduce(
      (text, [name, value]) => text.replaceAll(`{{${name}}}`, String(value)),
      copy,
    );
  };

  return {
    useTranslation: () => ({ t, i18n: { language: "ru", resolvedLanguage: "ru" } }),
  };
});

describe("UsersTable Russian localization", () => {
  it("shows Russian headers and empty state when Russian is selected", () => {
    render(
      <UsersTable
        data={[]}
        rowCount={0}
        isLoading={false}
        possibleUIRoles={{}}
        teams={[]}
        sorting={[]}
        onSortingChange={vi.fn()}
        pagination={{ pageIndex: 0, pageSize: 25 }}
        onPaginationChange={vi.fn()}
        columnFilters={[]}
        onColumnFiltersChange={vi.fn()}
        searchValue=""
        onSearchChange={vi.fn()}
        selectionEnabled={false}
        rowSelection={{}}
        onRowSelectionChange={vi.fn()}
        onUserClick={vi.fn()}
        onDeleteUser={vi.fn()}
        onResetPassword={vi.fn()}
      />,
    );

    expect(screen.getByText("ID пользователя")).toBeInTheDocument();
    expect(screen.getByText("Роль в прокси")).toBeInTheDocument();
    expect(screen.getByText("Пользователи не найдены")).toBeInTheDocument();
  });
});
