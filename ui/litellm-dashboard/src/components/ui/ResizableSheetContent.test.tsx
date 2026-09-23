import { describe, it, expect, beforeEach } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { Sheet, SheetTitle } from "@/components/ui/sheet";
import { ResizableSheetContent } from "./ResizableSheetContent";

function sheetContent(): HTMLElement {
  const el = document.querySelector('[data-slot="sheet-content"]');
  if (!el) throw new Error("no sheet content");
  return el as HTMLElement;
}

describe("ResizableSheetContent", () => {
  beforeEach(() => {
    localStorage.clear();
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
});
