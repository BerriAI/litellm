import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { DataTable } from "@/components/shared/DataTable";
import { getModelHubTableColumns, ModelHubData } from "./ModelHubTableColumns";

const mockModel: ModelHubData = {
  model_group: "gpt-4o",
  providers: ["openai", "azure", "bedrock"],
  max_input_tokens: 128000,
  max_output_tokens: 16384,
  input_cost_per_token: 0.0000025,
  output_cost_per_token: 0.00001,
  mode: "chat",
  supports_parallel_function_calling: false,
  supports_vision: true,
  supports_function_calling: true,
  is_public_model_group: true,
};

function renderTable(data: ModelHubData[], onModelClick = vi.fn()) {
  render(
    <DataTable
      data={data}
      columns={getModelHubTableColumns({ onModelClick })}
      getRowId={(model, index) => model.model_group || String(index)}
      sortingMode="client"
      size="compact"
    />,
  );
  return onModelClick;
}

describe("getModelHubTableColumns", () => {
  it("renders the model row", () => {
    renderTable([mockModel]);
    expect(screen.getByText("gpt-4o")).toBeInTheDocument();
  });

  it("shows the first two providers as display names and '+1' for overflow", () => {
    renderTable([mockModel]);
    expect(screen.getByText("OpenAI")).toBeInTheDocument();
    expect(screen.getByText("Azure")).toBeInTheDocument();
    expect(screen.queryByText("Amazon Bedrock")).not.toBeInTheDocument();
    expect(screen.getByText("+1")).toBeInTheDocument();
  });

  it("formats token limits and per-million costs", () => {
    renderTable([mockModel]);
    expect(screen.getByText("128.0K / 16.4K")).toBeInTheDocument();
    expect(screen.getByText("$2.50")).toBeInTheDocument();
    expect(screen.getByText("$10.00")).toBeInTheDocument();
  });

  it("shows capability badges only for supported features", () => {
    renderTable([mockModel]);
    expect(screen.getByText("Vision")).toBeInTheDocument();
    expect(screen.getByText("Function Calling")).toBeInTheDocument();
    expect(screen.queryByText("Parallel Function Calling")).not.toBeInTheDocument();
  });

  it("shows the public status badge", () => {
    renderTable([mockModel]);
    expect(screen.getByText("Yes")).toBeInTheDocument();
    renderTable([{ ...mockModel, model_group: "private-model", is_public_model_group: false }]);
    expect(screen.getByText("No")).toBeInTheDocument();
  });

  it("opens the model details when the name is clicked", async () => {
    const user = userEvent.setup();
    const onModelClick = renderTable([mockModel]);
    await user.click(screen.getByRole("button", { name: "gpt-4o" }));
    expect(onModelClick).toHaveBeenCalledWith(mockModel);
  });

  it("opens the model details from the actions menu", async () => {
    const user = userEvent.setup();
    const onModelClick = renderTable([mockModel]);
    await user.click(screen.getByTestId("model-hub-actions-gpt-4o"));
    await user.click(await screen.findByTestId("model-hub-action-details"));
    expect(onModelClick).toHaveBeenCalledWith(mockModel);
  });

  it("copies the model name from the actions menu", async () => {
    const user = userEvent.setup();
    renderTable([mockModel]);
    await user.click(screen.getByTestId("model-hub-actions-gpt-4o"));
    await user.click(await screen.findByTestId("model-hub-action-copy"));
    expect(await window.navigator.clipboard.readText()).toBe("gpt-4o");
  });
});
