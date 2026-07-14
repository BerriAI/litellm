import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CacheControlInjectionPointsSection from "./CacheControlInjectionPointsSection";

describe("CacheControlInjectionPointsSection", () => {
  it("should render the frontend-owned field without exposing the raw default params input", () => {
    const onChange = vi.fn();
    render(<CacheControlInjectionPointsSection value={{}} onChange={onChange} />);

    expect(screen.getByText("Cache Control Injection Points")).toBeInTheDocument();
    expect(screen.queryByText("Default LiteLLM Params")).not.toBeInTheDocument();
    expect(screen.queryByTestId("cache-control-location-select-0")).not.toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });

  it("should render persisted injection point fields", () => {
    render(
      <CacheControlInjectionPointsSection
        value={{ cache_control_injection_points: [{ location: "message", role: "system", index: -2 }] }}
        onChange={vi.fn()}
      />,
    );

    expect(screen.getByTestId("cache-control-location-select-0")).toBeInTheDocument();
    expect(screen.getByTestId("cache-control-index-input-0")).toHaveValue("-2");
  });

  it("should add injection points without replacing other default params", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<CacheControlInjectionPointsSection value={{ timeout: 30 }} onChange={onChange} />);

    await user.click(screen.getByRole("switch", { name: "Cache Control Injection Points" }));

    expect(onChange).toHaveBeenCalledWith({
      timeout: 30,
      cache_control_injection_points: [{ location: "message" }],
    });
  });

  it("should remove injection points without replacing other default params", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(
      <CacheControlInjectionPointsSection
        value={{ timeout: 30, cache_control_injection_points: [{ location: "message" }] }}
        onChange={onChange}
      />,
    );

    await user.click(screen.getByRole("switch", { name: "Cache Control Injection Points" }));

    expect(onChange).toHaveBeenCalledWith({ timeout: 30 });
  });

  it("should preserve unsupported injection point values", () => {
    const onChange = vi.fn();
    render(
      <CacheControlInjectionPointsSection
        value={{ cache_control_injection_points: [{ location: "tool_config" }] }}
        onChange={onChange}
      />,
    );

    expect(screen.getByRole("switch", { name: "Cache Control Injection Points" })).toBeDisabled();
    expect(screen.getByText(/will be preserved unchanged/i)).toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });
});
