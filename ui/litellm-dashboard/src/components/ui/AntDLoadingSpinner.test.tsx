import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

// Mock antd Spin component
vi.mock("antd", () => ({
  Spin: ({ indicator, size, ...props }: any) => (
    <div data-testid="spin" data-size={size} {...props}>
      {indicator}
    </div>
  ),
}));

// Mock the icon
vi.mock("@ant-design/icons", () => ({
  LoadingOutlined: ({ style, spin, ...props }: any) => (
    <span
      data-testid="loading-icon"
      data-spin={spin}
      style={style}
      {...props}
    />
  ),
}));

import { AntDLoadingSpinner } from "./AntDLoadingSpinner";

describe("AntDLoadingSpinner", () => {
  it("renders without props", () => {
    render(<AntDLoadingSpinner />);
    expect(screen.getByTestId("spin")).toBeInTheDocument();
    expect(screen.getByTestId("loading-icon")).toBeInTheDocument();
  });

  it("passes size prop to Spin", () => {
    render(<AntDLoadingSpinner size="large" />);
    expect(screen.getByTestId("spin")).toHaveAttribute("data-size", "large");
  });

  it("passes small size to Spin", () => {
    render(<AntDLoadingSpinner size="small" />);
    expect(screen.getByTestId("spin")).toHaveAttribute("data-size", "small");
  });

  it("applies custom fontSize to the icon", () => {
    render(<AntDLoadingSpinner fontSize={32} />);
    const icon = screen.getByTestId("loading-icon");
    expect(icon).toHaveStyle({ fontSize: "32px" });
  });

  it("does not set style when fontSize is not provided", () => {
    render(<AntDLoadingSpinner />);
    const icon = screen.getByTestId("loading-icon");
    expect(icon.style.fontSize).toBe("");
  });

  it("sets spin attribute on icon", () => {
    render(<AntDLoadingSpinner />);
    const icon = screen.getByTestId("loading-icon");
    expect(icon).toHaveAttribute("data-spin", "true");
  });
});
