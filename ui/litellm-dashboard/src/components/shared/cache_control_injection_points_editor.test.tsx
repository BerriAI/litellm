import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CacheControlInjectionPointsEditor from "./cache_control_injection_points_editor";

describe("CacheControlInjectionPointsEditor", () => {
  it("should render one row per point", () => {
    render(
      <CacheControlInjectionPointsEditor
        value={[
          { location: "message", role: "system" },
          { location: "message", index: 0 },
        ]}
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByTestId("cache-control-location-select-0")).toBeInTheDocument();
    expect(screen.getByTestId("cache-control-location-select-1")).toBeInTheDocument();
  });

  it("should render a single default row when value is empty", () => {
    render(<CacheControlInjectionPointsEditor value={[]} onChange={vi.fn()} />);
    expect(screen.getByTestId("cache-control-location-select-0")).toBeInTheDocument();
    expect(screen.queryByTestId("cache-control-location-select-1")).not.toBeInTheDocument();
  });

  it("should add a new row with the Add Injection Point button", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<CacheControlInjectionPointsEditor value={[{ location: "message" }]} onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: /add injection point/i }));

    expect(onChange).toHaveBeenCalledWith([{ location: "message" }, { location: "message" }]);
  });

  it("should not render a remove button when there is only one row", () => {
    render(<CacheControlInjectionPointsEditor value={[{ location: "message" }]} onChange={vi.fn()} />);
    expect(document.querySelector(".anticon-minus-circle")).not.toBeInTheDocument();
  });

  it("should remove a row when its remove icon is clicked", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(
      <CacheControlInjectionPointsEditor
        value={[
          { location: "message", role: "system" },
          { location: "message", role: "user" },
        ]}
        onChange={onChange}
      />,
    );

    const removeButtons = screen.getAllByRole("button", { name: /remove injection point/i });
    expect(removeButtons).toHaveLength(2);
    await user.click(removeButtons[0]);

    expect(onChange).toHaveBeenCalledWith([{ location: "message", role: "user" }]);
  });

  it("should accept an integer index", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    const ControlledEditor = () => {
      const [points, setPoints] = React.useState([{ location: "message" as const }]);
      const handleChange = (updatedPoints: typeof points) => {
        setPoints(updatedPoints);
        onChange(updatedPoints);
      };
      return <CacheControlInjectionPointsEditor value={points} onChange={handleChange} />;
    };

    render(<ControlledEditor />);

    const indexInput = screen.getByTestId("cache-control-index-input-0");
    await user.type(indexInput, "-1");

    expect(indexInput).toHaveValue("-1");
    expect(onChange).toHaveBeenLastCalledWith([{ location: "message", index: -1 }]);
  });

  it("should increment the index with the stepper", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<CacheControlInjectionPointsEditor value={[{ location: "message", index: 0 }]} onChange={onChange} />);

    const incrementButton = document.querySelector(".ant-input-number-handler-up");
    expect(incrementButton).not.toBeNull();
    await user.click(incrementButton as HTMLElement);

    expect(onChange).toHaveBeenLastCalledWith([{ location: "message", index: 1 }]);
  });
});
