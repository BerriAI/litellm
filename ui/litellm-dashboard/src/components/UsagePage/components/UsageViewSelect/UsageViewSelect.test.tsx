import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { UsageViewSelect } from "./UsageViewSelect";

vi.mock("antd", async () => {
  const React = await import("react");

  function Select(props: any) {
    const { value, onChange, options, ...rest } = props;
    return React.createElement(
      "select",
      {
        ...rest,
        value,
        onChange: (e: any) => onChange?.(e.target.value),
        role: "combobox",
      },
      options?.map((opt: any) => React.createElement("option", { key: opt.value, value: opt.value }, opt.label)),
    );
  }
  (Select as any).displayName = "AntdSelect";

  return { Select };
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
});
