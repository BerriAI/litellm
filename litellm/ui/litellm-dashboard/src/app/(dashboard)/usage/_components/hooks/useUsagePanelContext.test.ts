import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { useUsagePanelContext } from "./useUsagePanelContext";

const dashboards: HTMLDivElement[] = [];

function createDashboard() {
  const viewport = document.createElement("div");
  const scroller = document.createElement("div");
  const page = document.createElement("div");
  const trigger = document.createElement("button");
  const panelControl = document.createElement("button");
  trigger.textContent = "View Claude Desktop usage";
  panelControl.textContent = "Back to Usage";
  page.append(trigger);
  scroller.append(page);
  viewport.append(scroller, panelControl);
  document.body.append(viewport);
  dashboards.push(viewport);
  return { viewport, scroller, page, trigger, panelControl };
}

afterEach(() => {
  dashboards.splice(0).forEach((dashboard) => dashboard.remove());
});

describe("useUsagePanelContext", () => {
  it("should preserve nested scroll positions while open and return focus to the trigger on close", async () => {
    const { viewport, scroller, page, trigger, panelControl } = createDashboard();
    const { result, rerender } = renderHook(({ open }) => useUsagePanelContext(open), {
      initialProps: { open: false },
    });
    act(() => {
      result.current.pageRef.current = page;
      viewport.scrollTop = 160;
      scroller.scrollTop = 840;
      scroller.scrollLeft = 45;
      trigger.focus();
      result.current.remember();
      panelControl.focus();
      viewport.scrollTop = 0;
      scroller.scrollTop = 0;
      scroller.scrollLeft = 0;
    });

    rerender({ open: true });
    await waitFor(() => expect(scroller.scrollTop).toBe(840));
    expect(viewport.scrollTop).toBe(160);
    expect(scroller.scrollLeft).toBe(45);
    expect(panelControl).toHaveFocus();

    scroller.scrollTop = 12;
    viewport.scrollTop = 0;
    rerender({ open: false });
    await waitFor(() => expect(trigger).toHaveFocus());
    expect(viewport.scrollTop).toBe(160);
    expect(scroller.scrollTop).toBe(840);
    expect(scroller.scrollLeft).toBe(45);
  });

  it("should restore the new position when a user reopens the panel after scrolling", async () => {
    const { scroller, page, trigger, panelControl } = createDashboard();
    const { result, rerender } = renderHook(({ open }) => useUsagePanelContext(open), {
      initialProps: { open: false },
    });
    act(() => {
      result.current.pageRef.current = page;
    });

    for (const position of [250, 930]) {
      act(() => {
        scroller.scrollTop = position;
        trigger.focus();
        result.current.remember();
        panelControl.focus();
        scroller.scrollTop = 0;
      });
      rerender({ open: true });
      await waitFor(() => expect(scroller.scrollTop).toBe(position));
      rerender({ open: false });
      await waitFor(() => expect(trigger).toHaveFocus());
      expect(scroller.scrollTop).toBe(position);
    }
  });
});
