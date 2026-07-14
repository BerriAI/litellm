import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import OptionalPreCallChecksSelector from "./OptionalPreCallChecksSelector";

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>();
  return {
    ...actual,
    Select: ({ value, onChange, options, "data-testid": testId }: any) => (
      <select
        multiple
        data-testid={testId}
        value={value ?? []}
        onChange={(e) => onChange(Array.from(e.target.selectedOptions).map((o: any) => o.value))}
      >
        {(options || []).map((option: any) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    ),
  };
});

const baseMetadata = {
  optional_pre_call_checks: {
    ui_field_name: "Optional Pre-call Checks",
    field_description: "Extra checks the router runs before picking a deployment",
    link: null,
  },
};

const options = ["prompt_caching", "router_budget_limiting", "session_affinity"];

describe("OptionalPreCallChecksSelector", () => {
  it("should render one option per entry in options", () => {
    render(
      <OptionalPreCallChecksSelector value={[]} options={options} routerFieldsMetadata={{}} onChange={vi.fn()} />,
    );
    const select = screen.getByTestId("optional-pre-call-checks-select") as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => o.value)).toEqual(options);
  });

  it("should display default label when no metadata is provided", () => {
    render(
      <OptionalPreCallChecksSelector value={[]} options={options} routerFieldsMetadata={{}} onChange={vi.fn()} />,
    );
    expect(screen.getByText("Optional Pre-call Checks")).toBeInTheDocument();
  });

  it("should display the label and description from metadata when provided", () => {
    render(
      <OptionalPreCallChecksSelector
        value={[]}
        options={options}
        routerFieldsMetadata={baseMetadata}
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByText("Extra checks the router runs before picking a deployment")).toBeInTheDocument();
  });

  it("should render a Learn more link when metadata provides one", () => {
    const metadata = {
      optional_pre_call_checks: { ...baseMetadata.optional_pre_call_checks, link: "https://docs.example.com/checks" },
    };
    render(
      <OptionalPreCallChecksSelector value={[]} options={options} routerFieldsMetadata={metadata} onChange={vi.fn()} />,
    );
    const link = screen.getByRole("link", { name: /learn more/i });
    expect(link).toHaveAttribute("href", "https://docs.example.com/checks");
  });

  it("should call onChange with the selected checks", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(
      <OptionalPreCallChecksSelector value={[]} options={options} routerFieldsMetadata={{}} onChange={onChange} />,
    );

    await user.selectOptions(screen.getByTestId("optional-pre-call-checks-select"), "prompt_caching");

    expect(onChange).toHaveBeenCalledWith(["prompt_caching"]);
  });

  it("should reflect an already-selected value", () => {
    render(
      <OptionalPreCallChecksSelector
        value={["router_budget_limiting"]}
        options={options}
        routerFieldsMetadata={{}}
        onChange={vi.fn()}
      />,
    );
    const select = screen.getByTestId("optional-pre-call-checks-select") as HTMLSelectElement;
    const selected = Array.from(select.selectedOptions).map((o) => o.value);
    expect(selected).toEqual(["router_budget_limiting"]);
  });
});
