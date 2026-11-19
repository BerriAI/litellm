import { describe, it, expect, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { Sheet, SheetTitle } from "@/components/ui/sheet";
import { ResizableSheetContent } from "./ResizableSheetContent";

function sheetContent(): HTMLElement {
  return screen.getByRole("dialog");
}

describe("ResizableSheetContent", () => {
  beforeEach(() => {
    localStorage.clear();
    Object.defineProperty(window, "innerWidth", { value: 2000, configurable: true, writable: true });
  });

  it("resizes by dragging the handle, clamps to the minimum, and persists on release", () => {
    render(
      <Sheet open>
        <ResizableSheetContent storageKey="k">
          <SheetTitle>t</SheetTitle>
        </ResizableSheetContent>
      </Sheet>,
    );

    fireEvent.pointerDown(screen.getByRole("separator"));
    fireEvent.pointerMove(window, { clientX: window.innerWidth * 0.5 });
    expect(sheetContent().style.getPropertyValue("--sheet-width")).toBe("50%");

    fireEvent.pointerMove(window, { clientX: window.innerWidth * 0.9 });
    expect(sheetContent().style.getPropertyValue("--sheet-width")).toBe("40%");

    fireEvent.pointerUp(window);
    expect(localStorage.getItem("k")).toBe("40");
  });

  it("resizes from the keyboard and exposes the width via aria-valuenow", () => {
    render(
      <Sheet open>
        <ResizableSheetContent storageKey="k">
          <SheetTitle>t</SheetTitle>
        </ResizableSheetContent>
      </Sheet>,
    );

    const sep = screen.getByRole("separator");
    sep.focus();

    fireEvent.keyDown(sep, { key: "ArrowLeft" });
    expect(sheetContent().style.getPropertyValue("--sheet-width")).toBe("80%");
    expect(localStorage.getItem("k")).toBe("80");

    fireEvent.keyDown(sep, { key: "ArrowRight" });
    fireEvent.keyDown(sep, { key: "ArrowRight" });
    expect(sheetContent().style.getPropertyValue("--sheet-width")).toBe("70%");

    fireEvent.keyDown(sep, { key: "End" });
    expect(sheetContent().style.getPropertyValue("--sheet-width")).toBe("40%");
    expect(sep).toHaveAttribute("aria-valuenow", "40");
  });

  it("clamps the End key to the 720px floor when it exceeds the percentage minimum", () => {
    Object.defineProperty(window, "innerWidth", { value: 1000, configurable: true, writable: true });
    render(
      <Sheet open>
        <ResizableSheetContent storageKey="k">
          <SheetTitle>t</SheetTitle>
        </ResizableSheetContent>
      </Sheet>,
    );

    const sep = screen.getByRole("separator");
    fireEvent.keyDown(sep, { key: "End" });

    expect(sheetContent().style.getPropertyValue("--sheet-width")).toBe("72%");
    expect(sep).toHaveAttribute("aria-valuenow", "72");
  });

  it("announces the rendered width when the 720px floor overrides the stored percentage", () => {
    localStorage.setItem("k", "40");
    Object.defineProperty(window, "innerWidth", { value: 1000, configurable: true, writable: true });
    render(
      <Sheet open>
        <ResizableSheetContent storageKey="k">
          <SheetTitle>t</SheetTitle>
        </ResizableSheetContent>
      </Sheet>,
    );

    const sep = screen.getByRole("separator");
    expect(sep).toHaveAttribute("aria-valuenow", "72");
    expect(sep).toHaveAttribute("aria-valuemin", "72");

    Object.defineProperty(window, "innerWidth", { value: 2000, configurable: true, writable: true });
    fireEvent(window, new Event("resize"));
    expect(sep).toHaveAttribute("aria-valuenow", "40");
    expect(sep).toHaveAttribute("aria-valuemin", "40");

    Object.defineProperty(window, "innerWidth", { value: 800, configurable: true, writable: true });
    fireEvent(window, new Event("resize"));
    expect(sep).toHaveAttribute("aria-valuenow", "90");
    expect(localStorage.getItem("k")).toBe("40");
  });
});
