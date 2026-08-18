import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const ENVIRONMENTS = [
  { value: "development", label: "Development" },
  { value: "staging", label: "Staging" },
] as const;

function renderSelect(props: { value: string | null; items?: React.ComponentProps<typeof Select>["items"] }) {
  return render(
    <Select value={props.value} items={props.items}>
      <SelectTrigger data-testid="trigger">
        <SelectValue />
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
});
