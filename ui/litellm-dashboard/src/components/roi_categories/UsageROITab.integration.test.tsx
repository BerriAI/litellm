import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Tabs, TabsList } from "@/components/ui/tabs";
import { ROIUsageTrigger } from "./UsageROITab";

describe("ROI usage access", () => {
  it.each(["Admin", "proxy_admin"])("shows ROI for the %s administrator role", (role) => {
    render(
      <Tabs defaultValue="roi">
        <TabsList>
          <ROIUsageTrigger role={role} view="global" />
        </TabsList>
      </Tabs>,
    );
    expect(screen.getByRole("tab", { name: "ROI" })).toBeVisible();
  });

  it.each([
    ["Admin Viewer", "global"],
    ["proxy_admin_viewer", "global"],
    ["internal_user", "global"],
    ["proxy_admin", "team"],
  ])("hides ROI for %s in %s usage", (role, view) => {
    render(
      <Tabs defaultValue="roi">
        <TabsList>
          <ROIUsageTrigger role={role} view={view} />
        </TabsList>
      </Tabs>,
    );
    expect(screen.queryByRole("tab", { name: "ROI" })).not.toBeInTheDocument();
  });
});
