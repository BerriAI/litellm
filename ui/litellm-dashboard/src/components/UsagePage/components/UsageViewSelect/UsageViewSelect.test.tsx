import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { UsageViewSelect } from "./UsageViewSelect";

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
