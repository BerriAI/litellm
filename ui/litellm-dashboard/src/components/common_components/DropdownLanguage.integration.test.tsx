import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";
import TeamDropdown from "./team_dropdown";
import TeamMultiSelect from "./team_multi_select";
import UserDropdown from "./UserDropdown";

const { teamsQuery, usersQuery } = vi.hoisted(() => ({ teamsQuery: vi.fn(), usersQuery: vi.fn() }));
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({ useInfiniteTeams: teamsQuery }));
vi.mock("@/app/(dashboard)/hooks/users/useUsers", () => ({
  useInfiniteUsers: usersQuery,
  useUserLookup: () => ({ data: null }),
}));

const loadingQuery = () => ({
  data: undefined,
  isLoading: true,
  hasNextPage: false,
  isFetchingNextPage: false,
  fetchNextPage: vi.fn(),
});

// 查询边界返回加载状态，选择器及其真实子组件负责展示当前语言。
describe("分页团队与用户选择器语言", () => {
  beforeEach(() => {
    localStorage.clear();
    teamsQuery.mockReturnValue(loadingQuery());
    usersQuery.mockReturnValue(loadingQuery());
  });

  it.each([
    { kind: "team", chinese: "正在加载团队…", english: "Loading teams…" },
    { kind: "multi", chinese: "正在加载团队...", english: "Loading teams..." },
    { kind: "user", chinese: "正在加载用户…", english: "Loading users…" },
  ])("翻译 $kind 选择器加载提示且切换语言保留查询", async ({ kind, chinese, english }) => {
    const user = userEvent.setup();
    let control = <TeamDropdown />;
    if (kind === "multi") control = <TeamMultiSelect />;
    else if (kind === "user") control = <UserDropdown onChange={vi.fn()} />;
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="dropdown-language" storageScope="test">
        {control}
        <LanguageSelector />
      </LanguageProvider>,
    );
    const search = screen.getAllByRole("combobox").find((element) => element.tagName === "INPUT")!;
    await user.click(search);
    expect(await screen.findByText(chinese)).toBeVisible();
    await user.keyboard("{Escape}");
    fireEvent.change(await screen.findByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    await user.click(search);
    expect(await screen.findByText(english)).toBeVisible();
    expect(teamsQuery.mock.calls.every((args) => args[1] === undefined)).toBe(true);
    expect(usersQuery.mock.calls.every((args) => args[1] === undefined)).toBe(true);
  });

  it("仅翻译多选器默认placeholder和清除标签，保留调用方文本及团队别名", () => {
    teamsQuery.mockReturnValue({
      ...loadingQuery(),
      isLoading: false,
      data: { pages: [{ teams: [{ team_id: "team-1", team_alias: "Home" }] }] },
    });
    const { rerender } = render(
      <LanguageProvider defaultLanguage="zh-CN" userId="dropdown-language" storageScope="test">
        <TeamMultiSelect value={["team-1"]} />
      </LanguageProvider>,
    );
    expect(screen.getByPlaceholderText("按别名搜索团队…")).toBeVisible();
    expect(screen.getByRole("button", { name: "清除所有团队" })).toBeVisible();
    expect(screen.getByText("Home")).toBeVisible();
    rerender(
      <LanguageProvider defaultLanguage="zh-CN" userId="dropdown-language" storageScope="test">
        <TeamMultiSelect value={["team-1"]} placeholder="Settings" />
      </LanguageProvider>,
    );
    expect(screen.getByPlaceholderText("Settings")).toBeVisible();
    expect(screen.getByText("Home")).toBeVisible();
  });
});
