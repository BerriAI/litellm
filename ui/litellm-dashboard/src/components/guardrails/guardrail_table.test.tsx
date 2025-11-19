import GuardrailTable from "./guardrail_table";
import { render } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { GuardrailDefinitionLocation } from "./types";
describe("GuardrailTable", () => {
  it("should render", () => {
    const { getByText } = render(
      <GuardrailTable
        guardrailsList={[]}
        isLoading={false}
        onDeleteClick={() => {}}
        accessToken={null}
        onGuardrailUpdated={() => {}}
        onGuardrailClick={() => {}}
      />,
    );
    expect(getByText("Guardrail ID")).toBeInTheDocument();
    expect(getByText("Name")).toBeInTheDocument();
    expect(getByText("Provider")).toBeInTheDocument();
    expect(getByText("Mode")).toBeInTheDocument();
    expect(getByText("Default On")).toBeInTheDocument();
    expect(getByText("Created At")).toBeInTheDocument();
    expect(getByText("Updated At")).toBeInTheDocument();
  });

  it("should not allow deletion of config guardrails", () => {
    const { getByTestId } = render(
      <GuardrailTable
        guardrailsList={[
          {
            guardrail_id: "1",
            guardrail_name: "Guardrail 1",
            litellm_params: { guardrail: "presidio", mode: "pre_call", default_on: true },
            guardrail_info: null,
            created_at: "2021-01-01",
            updated_at: "2021-01-01",
            guardrail_definition_location: GuardrailDefinitionLocation.CONFIG,
          },
        ]}
        isLoading={false}
        onDeleteClick={() => {}}
        accessToken={null}
        onGuardrailUpdated={() => {}}
        onGuardrailClick={() => {}}
      />,
    );

    const deleteGuardrailButton = getByTestId("config-delete-icon");
    expect(deleteGuardrailButton).toBeInTheDocument();
    expect(deleteGuardrailButton).toHaveClass("cursor-not-allowed text-gray-400");
    expect(deleteGuardrailButton).toHaveAttribute(
      "title",
      "Config guardrail cannot be deleted on the dashboard. Please delete it from the config file.",
    );
  });
});
