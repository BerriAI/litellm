import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import OptionalPreCallChecksSelector, { OPTIONAL_PRE_CALL_CHECK_OPTIONS } from "./OptionalPreCallChecksSelector";

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

describe("OptionalPreCallChecksSelector", () => {
  it("should render the frontend-defined options", () => {
    render(<OptionalPreCallChecksSelector value={[]} onChange={vi.fn()} />);
    const select = screen.getByTestId("optional-pre-call-checks-select") as HTMLSelectElement;
    expect(Array.from(select.options).map((o) => o.value)).toEqual(OPTIONAL_PRE_CALL_CHECK_OPTIONS);
  });

  it("should display the frontend-defined label and description", () => {
    render(<OptionalPreCallChecksSelector value={[]} onChange={vi.fn()} />);
    expect(screen.getByText("Optional Pre-call Checks")).toBeInTheDocument();
    expect(screen.getByText(/extra checks the router runs before picking a deployment/i)).toBeInTheDocument();
  });

  it("should render the prompt caching documentation link", () => {
    render(<OptionalPreCallChecksSelector value={[]} onChange={vi.fn()} />);
    const link = screen.getByRole("link", { name: /learn more/i });
    expect(link).toHaveAttribute("href", "https://docs.litellm.ai/docs/tutorials/claude_code_prompt_cache_routing");
  });

  it("should call onChange with the selected checks", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<OptionalPreCallChecksSelector value={[]} onChange={onChange} />);

    await user.selectOptions(screen.getByTestId("optional-pre-call-checks-select"), "prompt_caching");

    expect(onChange).toHaveBeenCalledWith(["prompt_caching"]);
  });

  it("should reflect an already-selected value", () => {
    render(<OptionalPreCallChecksSelector value={["router_budget_limiting"]} onChange={vi.fn()} />);
    const select = screen.getByTestId("optional-pre-call-checks-select") as HTMLSelectElement;
    const selected = Array.from(select.selectedOptions).map((o) => o.value);
    expect(selected).toEqual(["router_budget_limiting"]);
  });
});
