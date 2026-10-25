import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { chooseSelectOption } from "@/../tests/test-utils";
import { UsageViewSelect } from "./UsageViewSelect";
import { LanguageProvider } from "@/contexts/LanguageContext";

const openMenu = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole("combobox"));
};

// The listbox is portalled outside the render container in both antd and Base UI, so an
// option is "offered" when the label appears more times on the page than inside the trigger.
const offers = (container: HTMLElement, label: string) =>
  screen.queryAllByText(label).length > within(container).queryAllByText(label).length;

describe("UsageViewSelect", () => {
  const mockOnChange = vi.fn();

  beforeEach(() => {
    mockOnChange.mockClear();
    localStorage.clear();
  });

  it("中文菜单保持角色可见性及原始用量路由值", async () => {
    const user = userEvent.setup();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="usage-test" storageScope="usage-test">
        <UsageViewSelect value="global" onChange={mockOnChange} userRole="Internal User" />
      </LanguageProvider>,
    );

    expect(screen.getByRole("heading", { name: "用量" })).toBeInTheDocument();
    const picker = screen.getByRole("combobox", { name: "选择要查看的用量数据" });
    await chooseSelectOption(user, picker, /^团队用量/);
    expect(mockOnChange).toHaveBeenCalledWith("team");
    await user.click(picker);
    expect(screen.queryByRole("option", { name: /^组织用量/ })).not.toBeInTheDocument();
  });

  it("should render", async () => {
    const user = userEvent.setup();
    const { container } = render(<UsageViewSelect value="global" onChange={mockOnChange} userRole="Internal User" />);

    expect(screen.getByRole("heading", { name: "Usage" })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Select the usage data you want to view" })).toBeInTheDocument();

    await openMenu(user);
    expect(offers(container, "Your Usage")).toBe(true);
  });

  it("should call onChange when value changes", async () => {
    const user = userEvent.setup();
    render(<UsageViewSelect value="global" onChange={mockOnChange} userRole="Admin" />);

    await chooseSelectOption(user, screen.getByRole("combobox"), /^Team Usage/);

    expect(mockOnChange).toHaveBeenCalled();
    expect(mockOnChange.mock.calls[0][0]).toBe("team");
  });

  it("should show Tag Usage for non-admin users with tag usage permission", async () => {
    const user = userEvent.setup();
    const { container } = render(
      <UsageViewSelect value="global" onChange={mockOnChange} userRole="Internal User" canViewTagUsage={true} />,
    );

    await openMenu(user);
    expect(offers(container, "Tag Usage")).toBe(true);
  });

  it("should hide Tag Usage for non-admin users without tag usage permission", async () => {
    const user = userEvent.setup();
    const { container } = render(<UsageViewSelect value="global" onChange={mockOnChange} userRole="Internal User" />);

    await openMenu(user);
    expect(offers(container, "Tag Usage")).toBe(false);
  });

  it.each(["Organization Usage", "Agent Usage (A2A)"])("should show %s to an admin", async (optionName) => {
    const user = userEvent.setup();
    const { container } = render(<UsageViewSelect value="global" onChange={mockOnChange} userRole="Admin" />);

    await openMenu(user);
    expect(offers(container, optionName)).toBe(true);
  });

  it.each(["Organization Usage", "Agent Usage (A2A)"])("should hide %s from an internal user", async (optionName) => {
    const user = userEvent.setup();
    const { container } = render(
      <UsageViewSelect value="global" onChange={mockOnChange} userRole="Internal User" canViewTagUsage={true} />,
    );

    await openMenu(user);
    expect(offers(container, optionName)).toBe(false);
  });

  // An org admin's session role is "Internal User" — org-admin-ness lives in the
  // membership table — so the two rows above cannot tell them apart from a plain
  // internal user. Organization Usage must open for them, and only that option:
  // the proxy serves them /organization/daily/activity scoped to the orgs they
  // administer, but still refuses the agent usage route.
  it.each([
    ["Organization Usage", true],
    ["Agent Usage (A2A)", false],
  ] as const)("should offer %s to an org admin: %s", async (optionName, expected) => {
    const user = userEvent.setup();
    const { container } = render(
      <UsageViewSelect value="global" onChange={mockOnChange} userRole="Internal User" isOrgAdmin={true} />,
    );

    await openMenu(user);
    expect(offers(container, optionName)).toBe(expected);
  });

  it.each(["Team Usage", "Tag Usage"])("should keep %s available to an internal user", async (optionName) => {
    const user = userEvent.setup();
    const { container } = render(
      <UsageViewSelect value="global" onChange={mockOnChange} userRole="Internal User" canViewTagUsage={true} />,
    );

    await openMenu(user);
    expect(offers(container, optionName)).toBe(true);
  });
});
