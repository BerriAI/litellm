import { describe, it, expect, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DefaultLitellmParamsSection from "./DefaultLitellmParamsSection";

describe("DefaultLitellmParamsSection", () => {
  it("should render the non-cache-control keys as JSON in the textarea", () => {
    render(<DefaultLitellmParamsSection value={{ timeout: 30, max_retries: 0 }} onChange={vi.fn()} />);
    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    expect(textarea.value).toContain('"timeout": 30');
    expect(textarea.value).toContain('"max_retries": 0');
  });

  it("should not show the cache control editor when no injection points are set", () => {
    render(<DefaultLitellmParamsSection value={{ timeout: 30 }} onChange={vi.fn()} />);
    expect(screen.queryByTestId("cache-control-location-select-0")).not.toBeInTheDocument();
  });

  it("should show the cache control editor pre-populated when injection points are already set", () => {
    render(
      <DefaultLitellmParamsSection
        value={{ cache_control_injection_points: [{ location: "message", role: "system" }] }}
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByTestId("cache-control-location-select-0")).toBeInTheDocument();
  });

  it("should preserve unsupported cache control points in the JSON editor", () => {
    render(
      <DefaultLitellmParamsSection
        value={{ cache_control_injection_points: [{ location: "tool_config" }] }}
        onChange={vi.fn()}
      />,
    );

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    expect(textarea.value).toContain('"location": "tool_config"');
    expect(screen.queryByTestId("cache-control-location-select-0")).not.toBeInTheDocument();
  });

  it("should call onChange with cache_control_injection_points added when the toggle is switched on", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<DefaultLitellmParamsSection value={{ timeout: 30 }} onChange={onChange} />);

    await user.click(screen.getByRole("switch"));

    expect(onChange).toHaveBeenCalledWith({ timeout: 30, cache_control_injection_points: [{ location: "message" }] });
  });

  it("should call onChange with cache_control_injection_points removed when the toggle is switched off", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(
      <DefaultLitellmParamsSection
        value={{ timeout: 30, cache_control_injection_points: [{ location: "message", role: "system" }] }}
        onChange={onChange}
      />,
    );

    await user.click(screen.getByRole("switch"));

    expect(onChange).toHaveBeenCalledWith({ timeout: 30 });
  });

  it("should merge edited JSON textarea content with cache_control_injection_points on blur", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(
      <DefaultLitellmParamsSection
        value={{ timeout: 30, cache_control_injection_points: [{ location: "message" }] }}
        onChange={onChange}
      />,
    );

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    await user.clear(textarea);
    await user.type(textarea, '{{"timeout": 60}');
    await user.tab();

    expect(onChange).toHaveBeenCalledWith({ timeout: 60, cache_control_injection_points: [{ location: "message" }] });
  });

  it("should not call onChange and should flag the field as invalid when the textarea has invalid JSON on blur", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<DefaultLitellmParamsSection value={{ timeout: 30 }} onChange={onChange} />);

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    await user.clear(textarea);
    await user.type(textarea, "not json");
    await user.tab();

    expect(onChange).not.toHaveBeenCalled();
    expect(textarea).toHaveClass("ant-input-status-error");
  });

  it("should clear the invalid state once the textarea is edited again", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<DefaultLitellmParamsSection value={{ timeout: 30 }} onChange={onChange} />);

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    await user.clear(textarea);
    await user.type(textarea, "not json");
    await user.tab();
    expect(textarea).toHaveClass("ant-input-status-error");

    await user.click(textarea);
    await user.type(textarea, "{{}}");

    expect(textarea).not.toHaveClass("ant-input-status-error");
  });

  it("should reject valid JSON that is not an object", async () => {
    const onChange = vi.fn();
    render(<DefaultLitellmParamsSection value={{ timeout: 30 }} onChange={onChange} />);

    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: "[]" } });
    fireEvent.blur(textarea);

    expect(onChange).not.toHaveBeenCalled();
    expect(textarea).toHaveClass("ant-input-status-error");
  });
});
