import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";

import { fireEvent, renderWithProviders, screen, waitFor } from "../../tests/test-utils";
import ModelFilters from "./model_filters";

type FilterableModel = ComponentProps<typeof ModelFilters>["modelHubData"][number];

const hubModel = (model_group: string, provider: string, mode: string): FilterableModel => ({
  model_group,
  providers: [provider],
  mode,
  supports_parallel_function_calling: false,
  supports_vision: false,
  supports_function_calling: false,
  is_public_model_group: false,
});

const ALPHA = hubModel("alpha", "anthropic", "chat");
const BETA = hubModel("beta", "openai", "embedding");
const GAMMA = hubModel("gamma", "openai", "chat");
const MODELS = [ALPHA, BETA, GAMMA];

const renderLocalFilters = () => {
  const onFilteredDataChange = vi.fn<(filteredData: FilterableModel[]) => void>();
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  renderWithProviders(<ModelFilters modelHubData={MODELS} onFilteredDataChange={onFilteredDataChange} />, {
    onUrlUpdate,
  });
  const lastFiltered = () => onFilteredDataChange.mock.calls.at(-1)?.[0].map((model) => model.model_group);
  return { onUrlUpdate, lastFiltered };
};

describe("ModelFilters without a filters state", () => {
  it("combines the filters it keeps locally, reports the matches and never touches the URL", async () => {
    const { onUrlUpdate, lastFiltered } = renderLocalFilters();
    await waitFor(() => expect(lastFiltered()).toEqual(["alpha", "beta", "gamma"]));

    fireEvent.change(screen.getByDisplayValue("All Providers"), { target: { value: "openai" } });
    await waitFor(() => expect(lastFiltered()).toEqual(["beta", "gamma"]));

    fireEvent.change(screen.getByDisplayValue("All Modes"), { target: { value: "chat" } });
    await waitFor(() => expect(lastFiltered()).toEqual(["gamma"]));
    expect(screen.getByDisplayValue("openai")).toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("Search model names..."), { target: { value: "zzz" } });
    await waitFor(() => expect(lastFiltered()).toEqual([]));

    expect(onUrlUpdate).not.toHaveBeenCalled();
  });

  it("clears every local filter and reports the full list again", async () => {
    const user = userEvent.setup();
    const { onUrlUpdate, lastFiltered } = renderLocalFilters();
    fireEvent.change(screen.getByDisplayValue("All Providers"), { target: { value: "openai" } });
    fireEvent.change(screen.getByDisplayValue("All Modes"), { target: { value: "chat" } });
    fireEvent.change(screen.getByPlaceholderText("Search model names..."), { target: { value: "gam" } });
    await waitFor(() => expect(lastFiltered()).toEqual(["gamma"]));

    await user.click(screen.getByRole("button", { name: "Clear Filters" }));

    await waitFor(() => expect(lastFiltered()).toEqual(["alpha", "beta", "gamma"]));
    expect(screen.getByPlaceholderText("Search model names...")).toHaveValue("");
    expect(screen.getByDisplayValue("All Providers")).toBeInTheDocument();
    expect(screen.getByDisplayValue("All Modes")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Clear Filters" })).not.toBeInTheDocument();
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });
});
