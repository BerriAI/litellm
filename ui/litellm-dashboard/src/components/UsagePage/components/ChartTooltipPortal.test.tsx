import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { beforeEach, describe, expect, it } from "vitest";
import { ChartTooltipPortal, useCursorPosition } from "./ChartTooltipPortal";

describe("ChartTooltipPortal", () => {
  beforeEach(() => {
    window.innerWidth = 1024;
    window.innerHeight = 768;
  });

  it("renders the tooltip on document.body so an overflow ancestor cannot clip it", () => {
    render(
      <div data-testid="clipping-container" style={{ overflow: "auto", maxHeight: 80 }}>
        <ChartTooltipPortal active position={{ x: 100, y: 100 }}>
          <span>Successful: 42</span>
        </ChartTooltipPortal>
      </div>,
    );

    const tooltip = screen.getByText("Successful: 42");
    expect(screen.getByTestId("clipping-container")).not.toContainElement(tooltip);
    expect(document.body).toContainElement(tooltip);
  });

  it("renders nothing when inactive", () => {
    render(
      <ChartTooltipPortal active={false} position={{ x: 0, y: 0 }}>
        <span>Successful: 42</span>
      </ChartTooltipPortal>,
    );

    expect(screen.queryByText("Successful: 42")).toBeNull();
  });

  it("pins the tooltip at the cursor with fixed positioning above other layers and ignores pointer events", () => {
    render(
      <ChartTooltipPortal active position={{ x: 120, y: 140 }}>
        <span>tip</span>
      </ChartTooltipPortal>,
    );

    const portal = screen.getByTestId("chart-tooltip-portal");
    expect(portal).toHaveStyle({ position: "fixed", pointerEvents: "none" });
    expect(portal.style.zIndex).toBe("9999");
    expect(portal.style.left).toBe("134px");
    expect(portal.style.top).toBe("154px");
    expect(portal.style.transform).toBe("translate(0, 0)");
  });

  it("flips toward the cursor near the right and bottom edges to stay on screen", () => {
    render(
      <ChartTooltipPortal active position={{ x: 1000, y: 700 }}>
        <span>tip</span>
      </ChartTooltipPortal>,
    );

    const portal = screen.getByTestId("chart-tooltip-portal");
    expect(portal.style.transform).toBe("translate(-100%, -100%)");
    expect(portal.style.left).toBe("986px");
    expect(portal.style.top).toBe("686px");
  });

  it("useCursorPosition records the latest cursor coordinates from mouse move", () => {
    function Harness() {
      const { positionRef, handleMouseMove } = useCursorPosition();
      const [, force] = useState(0);
      return (
        <div
          data-testid="area"
          onMouseMove={(event) => {
            handleMouseMove(event);
            force((tick) => tick + 1);
          }}
        >
          <ChartTooltipPortal active position={positionRef.current}>
            <span>tip</span>
          </ChartTooltipPortal>
        </div>
      );
    }

    render(<Harness />);
    fireEvent.mouseMove(screen.getByTestId("area"), { clientX: 220, clientY: 330 });

    const portal = screen.getByTestId("chart-tooltip-portal");
    expect(portal.style.left).toBe("234px");
    expect(portal.style.top).toBe("344px");
  });
});
