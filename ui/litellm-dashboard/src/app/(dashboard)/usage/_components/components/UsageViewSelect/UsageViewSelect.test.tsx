import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { chooseSelectOption } from "@/../tests/test-utils";
import { UsageViewSelect } from "./UsageViewSelect";

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
  });

  it("should render", async () => {
    const user = userEvent.setup();
    const { container } = render(<UsageViewSelect value="global" onChange={mockOnChange} userRole="Internal User" />);

    expect(screen.getByText("Usage View")).toBeInTheDocument();
    expect(screen.getByText("Select the usage data you want to view")).toBeInTheDocument();
    expect(screen.getByRole("combobox")).toBeInTheDocument();

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
