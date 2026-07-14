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

    const removeIcons = document.querySelectorAll(".anticon-minus-circle");
    expect(removeIcons).toHaveLength(2);
    await user.click(removeIcons[0] as HTMLElement);

    expect(onChange).toHaveBeenCalledWith([{ location: "message", role: "user" }]);
  });
});
