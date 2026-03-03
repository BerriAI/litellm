import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import TagFilteringToggle from "./TagFilteringToggle";

// setupTests.ts mocks @tremor/react but leaves Switch as the real implementation.
// Re-mock Switch as a plain checkbox so toggle interactions are trivially testable.
vi.mock("@tremor/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tremor/react")>();
  return {
    ...actual,
    Switch: ({ checked, onChange, className }: any) => (
      <input
        type="checkbox"
        role="switch"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className={className}
      />
    ),
  };
});

const baseMetadata = {
  enable_tag_filtering: {
    ui_field_name: "Tag Filtering",
    field_description: "Route requests based on tags",
    link: null,
  },
};

describe("TagFilteringToggle", () => {
  it("should render", () => {
    render(
      <TagFilteringToggle enabled={false} routerFieldsMetadata={{}} onToggle={vi.fn()} />
    );
    expect(screen.getByRole("switch")).toBeInTheDocument();
  });

  it("should display default label when no metadata is provided", () => {
    render(
      <TagFilteringToggle enabled={false} routerFieldsMetadata={{}} onToggle={vi.fn()} />
    );
    expect(screen.getByText("Enable Tag Filtering")).toBeInTheDocument();
  });

  it("should display the label from metadata when provided", () => {
    render(
      <TagFilteringToggle
        enabled={false}
        routerFieldsMetadata={baseMetadata}
        onToggle={vi.fn()}
      />
    );
    expect(screen.getByText("Tag Filtering")).toBeInTheDocument();
  });

  it("should display the description from metadata", () => {
    render(
      <TagFilteringToggle
        enabled={false}
        routerFieldsMetadata={baseMetadata}
        onToggle={vi.fn()}
      />
    );
    expect(screen.getByText("Route requests based on tags")).toBeInTheDocument();
  });

  it("should render a Learn more link when metadata provides one", () => {
    const metadata = {
      enable_tag_filtering: {
        ...baseMetadata.enable_tag_filtering,
        link: "https://docs.example.com/tag-filtering",
      },
    };
    render(
      <TagFilteringToggle enabled={false} routerFieldsMetadata={metadata} onToggle={vi.fn()} />
    );
    const link = screen.getByRole("link", { name: /learn more/i });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "https://docs.example.com/tag-filtering");
  });

  it("should not render a Learn more link when metadata has no link", () => {
    render(
      <TagFilteringToggle
        enabled={false}
        routerFieldsMetadata={baseMetadata}
        onToggle={vi.fn()}
      />
    );
    expect(screen.queryByRole("link", { name: /learn more/i })).not.toBeInTheDocument();
  });

  it("should reflect the enabled=true state on the switch", () => {
    render(
      <TagFilteringToggle enabled={true} routerFieldsMetadata={{}} onToggle={vi.fn()} />
    );
    expect(screen.getByRole("switch")).toBeChecked();
  });

  it("should call onToggle with the new value when the switch is toggled", async () => {
    const onToggle = vi.fn();
    const user = userEvent.setup();
    render(
      <TagFilteringToggle enabled={false} routerFieldsMetadata={{}} onToggle={onToggle} />
    );

    await user.click(screen.getByRole("switch"));

    expect(onToggle).toHaveBeenCalledWith(true);
  });
});
