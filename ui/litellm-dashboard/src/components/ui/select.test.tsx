import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const ENVIRONMENTS = [
  { value: "development", label: "Development" },
  { value: "staging", label: "Staging" },
] as const;

function renderSelect(props: {
  value: string | null;
  items?: React.ComponentProps<typeof Select>["items"];
  placeholder?: string;
}) {
  return render(
    <Select value={props.value} items={props.items}>
      <SelectTrigger data-testid="trigger">
        <SelectValue placeholder={props.placeholder} />
      </SelectTrigger>
      <SelectContent>
        {ENVIRONMENTS.map((environment) => (
          <SelectItem key={environment.value} value={environment.value}>
            {environment.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>,
  );
}

describe("SelectValue label resolution", () => {
  it("renders the raw value when the root carries no items", () => {
    renderSelect({ value: "development" });

    expect(screen.getByTestId("trigger")).toHaveTextContent("development");
  });

  it("renders the human label when the root carries items", () => {
    renderSelect({ value: "development", items: ENVIRONMENTS });

    expect(screen.getByTestId("trigger")).toHaveTextContent("Development");
  });

  it("falls back to the raw value for a value absent from items", () => {
    renderSelect({ value: "production", items: ENVIRONMENTS });

    expect(screen.getByTestId("trigger")).toHaveTextContent("production");
  });

  it("still renders the placeholder when nothing is selected", () => {
    renderSelect({ value: null, items: ENVIRONMENTS, placeholder: "Pick an environment" });

    expect(screen.getByTestId("trigger")).toHaveTextContent("Pick an environment");
  });

  it("lets a null-valued item's label win over the placeholder", () => {
    renderSelect({
      value: null,
      items: [{ value: null, label: "Any environment" }, ...ENVIRONMENTS],
      placeholder: "Pick an environment",
    });

    expect(screen.getByTestId("trigger")).toHaveTextContent("Any environment");
  });
});

function renderOpenableSelect(contentProps?: React.ComponentProps<typeof SelectContent>) {
  return render(
    <Select value={null} items={ENVIRONMENTS}>
      <SelectTrigger data-testid="trigger">
        <SelectValue placeholder="Pick an environment" />
      </SelectTrigger>
      <SelectContent data-testid="content" {...contentProps}>
        {ENVIRONMENTS.map((environment) => (
          <SelectItem key={environment.value} value={environment.value}>
            {environment.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>,
  );
}

describe("SelectContent anchoring", () => {
  it("anchors to the edge of the trigger rather than over it by default", async () => {
    const user = userEvent.setup();
    renderOpenableSelect();

    await user.click(screen.getByTestId("trigger"));

    expect(await screen.findByTestId("content")).toHaveAttribute("data-align-trigger", "false");
  });

  it("still lets a caller opt into item-aligned anchoring", async () => {
    const user = userEvent.setup();
    renderOpenableSelect({ alignItemWithTrigger: true });

    await user.click(screen.getByTestId("trigger"));

    expect(await screen.findByTestId("content")).toHaveAttribute("data-align-trigger", "true");
  });
});
