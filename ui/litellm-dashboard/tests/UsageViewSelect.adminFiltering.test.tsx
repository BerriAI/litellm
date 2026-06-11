import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, act } from "@testing-library/react";
import { UsageViewSelect } from "../src/components/UsagePage/components/UsageViewSelect/UsageViewSelect";

// ── Types for antd mocks ─────────────────────────────────────────────────────

type SelectOption = {
  value: string;
  label: React.ReactNode;
};

type SelectProps = {
  value?: string;
  onChange?: (value: string) => void;
  options?: SelectOption[];
};

type BadgeProps = {
  count?: React.ReactNode;
  children?: React.ReactNode;
};

// ── Mocks (mirrors the pattern from UsageViewSelect.test.tsx in src/) ──────────

vi.mock("antd", async () => {
  const React = await import("react");

  function Select(props: SelectProps) {
    const { value, onChange, options } = props;
    return React.createElement(
      "select",
      {
        value,
        onChange: (e: React.ChangeEvent<HTMLSelectElement>) => onChange?.(e.target.value),
        role: "combobox",
      },
      options?.map((opt) => React.createElement("option", { key: opt.value, value: opt.value }, opt.label)),
    );
  }
  Select.displayName = "AntdSelect";

  function Badge(props: BadgeProps) {
    return React.createElement("span", { "data-testid": "antd-badge" }, props.count, props.children);
  }
  Badge.displayName = "AntdBadge";

  return { Select, Badge };
});

vi.mock("@ant-design/icons", async () => {
  const React = await import("react");
  const Icon = () => React.createElement("span", { "data-testid": "icon" });
  return {
    GlobalOutlined: Icon,
    BankOutlined: Icon,
    TeamOutlined: Icon,
    ShoppingCartOutlined: Icon,
    TagsOutlined: Icon,
    RobotOutlined: Icon,
    UserOutlined: Icon,
    LineChartOutlined: Icon,
    BarChartOutlined: Icon,
  };
});

// ── Admin-only option values ───────────────────────────────────────────────────

const ADMIN_ONLY_VALUES = ["customer", "tag", "agent", "user", "user-agent-activity"];
const ALL_VALUES = ["global", "organization", "team", "customer", "tag", "agent", "user", "user-agent-activity"];
const NON_ADMIN_VISIBLE = ["global", "organization", "team"];

// ── Helpers ────────────────────────────────────────────────────────────────────

function getOptionValues(): string[] {
  return Array.from(screen.getByRole("combobox").querySelectorAll("option")).map((o) => (o as HTMLOptionElement).value);
}

// ── Tests ──────────────────────────────────────────────────────────────────────

describe("UsageViewSelect — admin vs non-admin option filtering", () => {
  const mockOnChange = vi.fn();

  beforeEach(() => {
    mockOnChange.mockClear();
  });

  it("should render without crashing", () => {
    render(<UsageViewSelect value="global" onChange={mockOnChange} isAdmin={false} />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("should expose all 8 options to admin users", () => {
    render(<UsageViewSelect value="global" onChange={mockOnChange} isAdmin={true} />);
    const values = getOptionValues();
    expect(values).toHaveLength(8);
    ALL_VALUES.forEach((v) => expect(values).toContain(v));
  });

  it("should hide adminOnly options from non-admin users", () => {
    render(<UsageViewSelect value="global" onChange={mockOnChange} isAdmin={false} />);
    const values = getOptionValues();
    ADMIN_ONLY_VALUES.forEach((v) => expect(values).not.toContain(v));
  });

  it("should show only 3 options (global, organization, team) for non-admin users", () => {
    render(<UsageViewSelect value="global" onChange={mockOnChange} isAdmin={false} />);
    const values = getOptionValues();
    expect(values).toHaveLength(NON_ADMIN_VISIBLE.length);
    NON_ADMIN_VISIBLE.forEach((v) => expect(values).toContain(v));
  });

  it("should show 'Your Usage' instead of 'Global Usage' for non-admin users", () => {
    render(<UsageViewSelect value="global" onChange={mockOnChange} isAdmin={false} />);
    const options = Array.from(screen.getByRole("combobox").querySelectorAll("option"));
    const globalOption = options.find((o) => (o as HTMLOptionElement).value === "global") as HTMLOptionElement;
    expect(globalOption.textContent).toBe("Your Usage");
  });

  it("should show 'Global Usage' for admin users", () => {
    render(<UsageViewSelect value="global" onChange={mockOnChange} isAdmin={true} />);
    const options = Array.from(screen.getByRole("combobox").querySelectorAll("option"));
    const globalOption = options.find((o) => (o as HTMLOptionElement).value === "global") as HTMLOptionElement;
    expect(globalOption.textContent).toBe("Global Usage");
  });

  it("should show 'Your Organization Usage' instead of 'Organization Usage' for non-admin users", () => {
    render(<UsageViewSelect value="organization" onChange={mockOnChange} isAdmin={false} />);
    const options = Array.from(screen.getByRole("combobox").querySelectorAll("option"));
    const orgOption = options.find((o) => (o as HTMLOptionElement).value === "organization") as HTMLOptionElement;
    expect(orgOption.textContent).toBe("Your Organization Usage");
  });

  it("should show 'Organization Usage' label for admin users", () => {
    render(<UsageViewSelect value="organization" onChange={mockOnChange} isAdmin={true} />);
    const options = Array.from(screen.getByRole("combobox").querySelectorAll("option"));
    const orgOption = options.find((o) => (o as HTMLOptionElement).value === "organization") as HTMLOptionElement;
    expect(orgOption.textContent).toBe("Organization Usage");
  });

  it("should keep 'Team Usage' label unchanged for both admin and non-admin", () => {
    render(<UsageViewSelect value="team" onChange={mockOnChange} isAdmin={false} />);
    const options = Array.from(screen.getByRole("combobox").querySelectorAll("option"));
    const teamOption = options.find((o) => (o as HTMLOptionElement).value === "team") as HTMLOptionElement;
    expect(teamOption.textContent).toBe("Team Usage");
  });

  it("should call onChange with the correct option value when user changes selection", () => {
    render(<UsageViewSelect value="global" onChange={mockOnChange} isAdmin={true} />);
    act(() => {
      fireEvent.change(screen.getByRole("combobox"), { target: { value: "team" } });
    });
    expect(mockOnChange).toHaveBeenCalledWith("team");
  });

  it("should use custom title and description when provided", () => {
    render(
      <UsageViewSelect
        value="global"
        onChange={mockOnChange}
        isAdmin={false}
        title="My Custom Title"
        description="My custom description"
      />,
    );
    expect(screen.getByText("My Custom Title")).toBeInTheDocument();
    expect(screen.getByText("My custom description")).toBeInTheDocument();
  });
});
