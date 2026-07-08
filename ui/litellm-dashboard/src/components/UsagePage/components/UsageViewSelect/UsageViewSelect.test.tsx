import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  getConfigurableNonAdminUsageViews,
  getVisibleUsageOptions,
  resolveActiveUsageView,
  UsageViewSelect,
} from "./UsageViewSelect";

vi.mock("antd", async () => {
  const React = await import("react");

  function Select(props: any) {
    const { value, onChange, options, optionRender, labelRender, ...rest } = props;
    const selectedOption = options?.find((opt: any) => opt.value === value);
    const renderedLabel = labelRender ? labelRender({ value, label: selectedOption?.label }) : selectedOption?.label;

    const optionElements = options?.map((opt: any) => {
      const rendered = optionRender ? optionRender({ value: opt.value, label: opt.label }) : opt.label;
      return React.createElement("option", { key: opt.value, value: opt.value }, opt.label);
    });

    const optionRenderOutputs = options
      ?.map((opt: any) => {
        if (optionRender) {
          const rendered = optionRender({ value: opt.value, label: opt.label });
          return React.createElement(
            "div",
            {
              key: `option-render-${opt.value}`,
              "data-testid": `option-render-${opt.value}`,
              style: { display: "none" },
            },
            rendered,
          );
        }
        return null;
      })
      .filter(Boolean);

    return React.createElement(
      React.Fragment,
      null,
      React.createElement(
        "select",
        {
          ...rest,
          value,
          onChange: (e: any) => onChange?.(e.target.value),
          role: "combobox",
        },
        optionElements,
      ),
      ...(optionRenderOutputs || []),
    );
  }
  (Select as any).displayName = "AntdSelect";

  function Badge(props: any) {
    const { count, color, children, ...rest } = props;
    return React.createElement(
      "span",
      { ...rest, "data-testid": "antd-badge", "data-color": color },
      count && React.createElement("span", { "data-testid": "antd-badge-count" }, count),
      children,
    );
  }
  (Badge as any).displayName = "AntdBadge";

  return { Select, Badge };
});

vi.mock("@ant-design/icons", async () => {
  const React = await import("react");

  function Icon(props: any) {
    return React.createElement("span", { "data-testid": "antd-icon" });
  }

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

describe("UsageViewSelect", () => {
  const mockOnChange = vi.fn();

  beforeEach(() => {
    mockOnChange.mockClear();
  });

  it("should render", () => {
    render(<UsageViewSelect value="global" onChange={mockOnChange} isAdmin={false} />);

    expect(screen.getByText("Usage View")).toBeInTheDocument();
    expect(screen.getByText("Select the usage data you want to view")).toBeInTheDocument();
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("should call onChange when value changes", () => {
    render(<UsageViewSelect value="global" onChange={mockOnChange} isAdmin={true} />);

    const select = screen.getByRole("combobox");
    act(() => {
      fireEvent.change(select, { target: { value: "team" } });
    });

    expect(mockOnChange).toHaveBeenCalledWith("team");
  });

  it("should show Tag Usage for non-admin users with tag usage permission", () => {
    render(<UsageViewSelect value="global" onChange={mockOnChange} isAdmin={false} canViewTagUsage={true} />);

    expect(screen.getByRole("option", { name: "Tag Usage" })).toBeInTheDocument();
  });

  it("should hide Tag Usage for non-admin users without tag usage permission", () => {
    render(<UsageViewSelect value="global" onChange={mockOnChange} isAdmin={false} />);

    expect(screen.queryByRole("option", { name: "Tag Usage" })).not.toBeInTheDocument();
  });

  it("should restrict non-admin options to enabledViews when set", () => {
    render(
      <UsageViewSelect
        value="global"
        onChange={mockOnChange}
        isAdmin={false}
        canViewTagUsage={true}
        enabledViews={["global"]}
      />,
    );

    expect(screen.getByRole("option", { name: "Your Usage" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "Your Organization Usage" })).not.toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "Team Usage" })).not.toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "Tag Usage" })).not.toBeInTheDocument();
  });

  it("should ignore enabledViews for admins", () => {
    render(<UsageViewSelect value="global" onChange={mockOnChange} isAdmin={true} enabledViews={["global"]} />);

    expect(screen.getByRole("option", { name: "Team Usage" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Organization Usage" })).toBeInTheDocument();
  });

  it("should show all non-admin options when enabledViews is null", () => {
    render(
      <UsageViewSelect
        value="global"
        onChange={mockOnChange}
        isAdmin={false}
        canViewTagUsage={true}
        enabledViews={null}
      />,
    );

    expect(screen.getByRole("option", { name: "Your Usage" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Your Organization Usage" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Team Usage" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Tag Usage" })).toBeInTheDocument();
  });
});

describe("getVisibleUsageOptions", () => {
  it("should return only role-visible non-admin views by default", () => {
    expect(getVisibleUsageOptions({ isAdmin: false, canViewTagUsage: true })).toEqual([
      "global",
      "organization",
      "team",
      "tag",
    ]);
  });

  it("should exclude tag when canViewTagUsage is false", () => {
    expect(getVisibleUsageOptions({ isAdmin: false, canViewTagUsage: false })).toEqual([
      "global",
      "organization",
      "team",
    ]);
  });

  it("should intersect with enabledViews for non-admins", () => {
    expect(getVisibleUsageOptions({ isAdmin: false, canViewTagUsage: true, enabledViews: ["global", "team"] })).toEqual(
      ["global", "team"],
    );
  });

  it("should ignore enabledViews for admins", () => {
    const adminViews = getVisibleUsageOptions({ isAdmin: true, enabledViews: ["global"] });
    expect(adminViews).toContain("team");
    expect(adminViews).toContain("tag");
    expect(adminViews).toContain("customer");
  });
});

describe("getConfigurableNonAdminUsageViews", () => {
  it("should list exactly the views configurable for non-admins with their non-admin labels", () => {
    expect(getConfigurableNonAdminUsageViews()).toEqual([
      { value: "global", label: "Your Usage", description: "View your usage" },
      { value: "organization", label: "Your Organization Usage", description: "View your organization's usage" },
      { value: "team", label: "Team Usage", description: "View usage by team" },
      { value: "tag", label: "Tag Usage", description: "View usage grouped by tags" },
    ]);
  });
});

describe("resolveActiveUsageView", () => {
  it("should keep the current view when it is still visible", () => {
    expect(resolveActiveUsageView("team", ["global", "team", "tag"])).toBe("team");
  });

  it("should fall back to the first visible view when the current one is hidden", () => {
    expect(resolveActiveUsageView("team", ["global", "tag"])).toBe("global");
  });

  it("should keep the current view when there are no visible views to fall back to", () => {
    expect(resolveActiveUsageView("team", [])).toBe("team");
  });
});
