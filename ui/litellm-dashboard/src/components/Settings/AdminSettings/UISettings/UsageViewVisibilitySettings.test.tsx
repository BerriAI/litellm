import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import UsageViewVisibilitySettings from "./UsageViewVisibilitySettings";

vi.mock("@/components/UsagePage/components/UsageViewSelect/UsageViewSelect", () => ({
  getConfigurableNonAdminUsageViews: () => [
    { value: "global", label: "Your Usage", description: "View your usage" },
    { value: "organization", label: "Your Organization Usage", description: "View your organization's usage" },
    { value: "team", label: "Team Usage", description: "View usage by team" },
    { value: "tag", label: "Tag Usage", description: "View usage grouped by tags" },
  ],
}));

describe("UsageViewVisibilitySettings", () => {
  it("should render the not-set tag when enabledViewsInternalUsers is null", () => {
    render(<UsageViewVisibilitySettings enabledViewsInternalUsers={null} isUpdating={false} onUpdate={vi.fn()} />);
    expect(screen.getByText("Not set (all views visible)")).toBeInTheDocument();
  });

  it("should show the selected view count tag when views are configured", () => {
    render(
      <UsageViewVisibilitySettings
        enabledViewsInternalUsers={["global", "team"]}
        isUpdating={false}
        onUpdate={vi.fn()}
      />,
    );
    expect(screen.getByText("2 views selected")).toBeInTheDocument();
  });

  it("should show singular 'view' when exactly one view is selected", () => {
    render(
      <UsageViewVisibilitySettings enabledViewsInternalUsers={["global"]} isUpdating={false} onUpdate={vi.fn()} />,
    );
    expect(screen.getByText("1 view selected")).toBeInTheDocument();
  });

  it("should save only the selected views", async () => {
    const onUpdate = vi.fn();
    const user = userEvent.setup();
    render(
      <UsageViewVisibilitySettings enabledViewsInternalUsers={["global"]} isUpdating={false} onUpdate={onUpdate} />,
    );

    await user.click(screen.getByRole("button", { name: /configure usage view visibility/i }));
    await user.click(await screen.findByRole("button", { name: /save usage view visibility settings/i }));

    expect(onUpdate).toHaveBeenCalledWith({ enabled_usage_views_internal_users: ["global"] });
  });

  it("should save null when no views are selected", async () => {
    const onUpdate = vi.fn();
    const user = userEvent.setup();
    render(
      <UsageViewVisibilitySettings enabledViewsInternalUsers={["global"]} isUpdating={false} onUpdate={onUpdate} />,
    );

    await user.click(screen.getByRole("button", { name: /configure usage view visibility/i }));
    await user.click(await screen.findByRole("checkbox", { name: /your usage/i }));
    await user.click(screen.getByRole("button", { name: /save usage view visibility settings/i }));

    expect(onUpdate).toHaveBeenCalledWith({ enabled_usage_views_internal_users: null });
  });

  it("should call onUpdate with null when reset button is clicked", async () => {
    const onUpdate = vi.fn();
    const user = userEvent.setup();
    render(<UsageViewVisibilitySettings enabledViewsInternalUsers={["team"]} isUpdating={false} onUpdate={onUpdate} />);

    await user.click(screen.getByRole("button", { name: /configure usage view visibility/i }));
    await user.click(await screen.findByRole("button", { name: /reset to default/i }));

    expect(onUpdate).toHaveBeenCalledWith({ enabled_usage_views_internal_users: null });
  });

  it("should display the property description when provided", () => {
    render(
      <UsageViewVisibilitySettings
        enabledViewsInternalUsers={null}
        enabledViewsPropertyDescription="Controls which usage views are visible"
        isUpdating={false}
        onUpdate={vi.fn()}
      />,
    );
    expect(screen.getByText("Controls which usage views are visible")).toBeInTheDocument();
  });
});
