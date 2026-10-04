import { act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { I18nextProvider } from "react-i18next";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createDashboardI18n } from "@/i18n";
import { renderWithProviders, screen } from "../../../../tests/test-utils";
import { BlogDropdown } from "./BlogDropdown";

const mockRefetch = vi.fn();
let mockIsError = false;

vi.mock("@/app/(dashboard)/hooks/useDisableBlogPosts", () => ({
  useDisableBlogPosts: () => false,
}));

vi.mock("@/app/(dashboard)/hooks/blogPosts/useBlogPosts", () => ({
  useBlogPosts: () => ({
    data: {
      posts: [{ title: "Post One", date: "2026-02-01", description: "Description one", url: "https://example.com/1" }],
    },
    isLoading: false,
    isError: mockIsError,
    refetch: mockRefetch,
  }),
}));

async function openDropdown() {
  await userEvent.setup().hover(screen.getByRole("button", { name: "Blog" }));
}

describe("blog dropdown language changes", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockIsError = false;
  });

  it("should translate Retry after switching languages and still refetch posts", async () => {
    mockIsError = true;
    const user = userEvent.setup();
    const i18n = createDashboardI18n();
    renderWithProviders(
      <I18nextProvider i18n={i18n}>
        <BlogDropdown />
      </I18nextProvider>,
    );

    await openDropdown();
    expect(await screen.findByRole("button", { name: "Retry" })).toBeInTheDocument();
    await act(async () => {
      await i18n.changeLanguage("zh-CN");
    });
    expect(screen.getByText("加载文章失败")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "重试" }));
    expect(mockRefetch).toHaveBeenCalledTimes(1);
  });

  it("should update visible dates when switching between English and Chinese", async () => {
    const i18n = createDashboardI18n();
    renderWithProviders(
      <I18nextProvider i18n={i18n}>
        <BlogDropdown />
      </I18nextProvider>,
    );

    await openDropdown();
    expect(await screen.findByText("Feb 1, 2026")).toBeInTheDocument();
    await act(async () => {
      await i18n.changeLanguage("zh-CN");
    });
    expect(screen.getByText("2026年2月1日")).toBeInTheDocument();
    expect(screen.queryByText("Feb 1, 2026")).not.toBeInTheDocument();
    expect(screen.getByText("Post One")).toBeInTheDocument();
    await act(async () => {
      await i18n.changeLanguage("en");
    });
    expect(screen.getByText("Feb 1, 2026")).toBeInTheDocument();
    expect(screen.queryByText("2026年2月1日")).not.toBeInTheDocument();
  });
});
