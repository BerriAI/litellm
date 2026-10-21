import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, renderWithProviders, screen } from "../../../tests/test-utils";
import { MODEL_MAX_BUDGET_PREMIUM_HINT, ModelMaxBudgetEditor, type ModelMaxBudget } from "./ModelMaxBudgetEditor";

const STORED: ModelMaxBudget = { "gpt-4o": { budget_limit: 5, time_period: "30d" } };

const renderEditor = (premiumUser: boolean, value: ModelMaxBudget = STORED) =>
  renderWithProviders(
    <ModelMaxBudgetEditor
      value={value}
      onChange={vi.fn()}
      availableModels={["gpt-4o", "claude-opus-4-8"]}
      premiumUser={premiumUser}
    />,
  );

const addButton = () => screen.getByRole("button", { name: /Add Model Budget/i });

// The proxy refuses a populated model_max_budget without an enterprise license,
// so an editable field would only ever hand a non-premium operator a 400 after
// they had filled the whole form in.
describe("ModelMaxBudgetEditor without an enterprise license", () => {
  it("locks every control on an existing row", () => {
    renderEditor(false);

    expect(screen.getByPlaceholderText("Max spend ($)")).toBeDisabled();
    expect(addButton()).toBeDisabled();
  });

  it("still shows the budgets already stored, so they stay auditable", () => {
    renderEditor(false);

    expect(screen.getByPlaceholderText("Max spend ($)")).toHaveValue(5);
  });

  it("says why the controls are locked instead of failing silently", () => {
    renderEditor(false);

    expect(screen.getByText(MODEL_MAX_BUDGET_PREMIUM_HINT)).toBeInTheDocument();
  });

  it("locks the empty state too, so no row can be started", () => {
    renderEditor(false, {});

    expect(addButton()).toBeDisabled();
    expect(screen.getByText(MODEL_MAX_BUDGET_PREMIUM_HINT)).toBeInTheDocument();
  });
});

describe("ModelMaxBudgetEditor with an enterprise license", () => {
  it("leaves every control usable", () => {
    renderEditor(true);

    expect(screen.getByPlaceholderText("Max spend ($)")).toBeEnabled();
    expect(addButton()).toBeEnabled();
  });

  it("does not tell a licensed operator to upgrade", () => {
    renderEditor(true);

    expect(screen.queryByText(MODEL_MAX_BUDGET_PREMIUM_HINT)).not.toBeInTheDocument();
  });

  it("leaves the empty state usable", () => {
    renderEditor(true, {});

    expect(addButton()).toBeEnabled();
  });
});

describe("ModelMaxBudgetEditor model families", () => {
  it("stores a typed pattern that matches no listed model as one budget entry", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithProviders(
      <ModelMaxBudgetEditor
        value={{}}
        onChange={onChange}
        availableModels={["gpt-4o", "claude-opus-4-8"]}
        premiumUser={true}
      />,
    );

    await user.click(addButton());
    const modelField = screen.getByPlaceholderText("Select model");
    await user.type(modelField, "claude-opus*");
    await user.click(await screen.findByRole("option", { name: /claude-opus\*/ }));
    fireEvent.change(screen.getByPlaceholderText("Max spend ($)"), { target: { value: "10" } });

    expect(onChange).toHaveBeenLastCalledWith({ "claude-opus*": { budget_limit: 10, time_period: "30d" } });
  });

  it("does not offer a typed pattern that another row already budgets", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ModelMaxBudgetEditor
        value={{ "claude-opus*": { budget_limit: 10, time_period: "1d" } }}
        onChange={vi.fn()}
        availableModels={["gpt-4o", "claude-opus-4-8"]}
        premiumUser={true}
      />,
    );

    await user.click(addButton());
    const [, newRowModelField] = screen.getAllByPlaceholderText("Select model");
    await user.type(newRowModelField, "claude-opus*");

    expect(screen.queryByRole("option", { name: /claude-opus\*/ })).not.toBeInTheDocument();
    await user.clear(newRowModelField);
    await user.type(newRowModelField, "claude-sonnet*");
    expect(await screen.findByRole("option", { name: /claude-sonnet\*/ })).toBeInTheDocument();
  });
});
