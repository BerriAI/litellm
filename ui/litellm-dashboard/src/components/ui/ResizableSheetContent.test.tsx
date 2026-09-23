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
});
