import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen, waitFor } from "../../../../../../tests/test-utils";
import { useZodForm } from "@/lib/forms/useZodForm";
import { ProjectBaseForm } from "./ProjectBaseForm";
import { emptyProjectFormValues, projectFormSchema } from "./projectFormSchema";

const mockUseTeams = vi.fn();
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeams: () => mockUseTeams(),
}));

vi.mock("@/components/organisms/create_key_button", () => ({
  fetchTeamModels: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/components/networking", () => ({
  getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [] }),
}));

vi.mock("@/components/key_team_helpers/fetch_available_models_team_key", () => ({
  getModelDisplayName: (model: string) => model,
}));

function FormWrapper() {
  const [advancedOpen, setAdvancedOpen] = React.useState(false);
  const form = useZodForm(projectFormSchema, { defaultValues: emptyProjectFormValues });
  return <ProjectBaseForm form={form} advancedOpen={advancedOpen} onAdvancedOpenChange={setAdvancedOpen} />;
}

describe("ProjectBaseForm", () => {
  beforeEach(() => {
    mockUseTeams.mockReturnValue({ data: [], isLoading: false });
  });

  it("should render", () => {
    renderWithProviders(<FormWrapper />);
    expect(screen.getByLabelText("Project Name")).toBeInTheDocument();
  });

  it("should show a 'Basic Information' section heading", () => {
    renderWithProviders(<FormWrapper />);
    expect(screen.getByText("Basic Information")).toBeInTheDocument();
  });

  it("should show a Project Name input", () => {
    renderWithProviders(<FormWrapper />);
    expect(screen.getByPlaceholderText("e.g. Customer Support Bot")).toBeInTheDocument();
  });

  it("should show a Team select", () => {
    renderWithProviders(<FormWrapper />);
    expect(screen.getByText("Team")).toBeInTheDocument();
  });

  it("should show a Description textarea", () => {
    renderWithProviders(<FormWrapper />);
    expect(screen.getByPlaceholderText("Describe the purpose of this project")).toBeInTheDocument();
  });

  it("should show the models select as disabled when no team is selected", () => {
    renderWithProviders(<FormWrapper />);
    // The models select should be disabled — its placeholder indicates no team yet
    expect(screen.getByText("Select a team first")).toBeInTheDocument();
  });

  it("should show available team options when the Team dropdown is opened", async () => {
    const user = userEvent.setup();
    mockUseTeams.mockReturnValue({
      data: [
        { team_id: "team-1", team_alias: "Engineering", models: [] },
        { team_id: "team-2", team_alias: "Sales", models: [] },
      ],
      isLoading: false,
    });
    renderWithProviders(<FormWrapper />);
    // The form label "Team" is associated with the combobox input inside the Select
    await user.click(screen.getByLabelText("Team"));
    await waitFor(() => {
      expect(screen.getByText("Engineering")).toBeInTheDocument();
    });
    expect(screen.getByText("Sales")).toBeInTheDocument();
  });

  it("should show the Max Budget field", () => {
    renderWithProviders(<FormWrapper />);
    expect(screen.getByPlaceholderText("0.00")).toBeInTheDocument();
  });

  it("should show the Advanced Settings collapse panel", () => {
    renderWithProviders(<FormWrapper />);
    expect(screen.getByText("Advanced Settings")).toBeInTheDocument();
  });

  it("should show a Guardrails field in the Advanced Settings section", async () => {
    const user = userEvent.setup();
    renderWithProviders(<FormWrapper />);
    await user.click(screen.getByText("Advanced Settings"));
    await waitFor(() => {
      expect(screen.getByText("Guardrails")).toBeInTheDocument();
    });
  });

  it("should show combined, input, and output TPM limit inputs for a model row", async () => {
    const user = userEvent.setup();
    renderWithProviders(<FormWrapper />);
    await user.click(screen.getByText("Advanced Settings"));
    await user.click(screen.getByRole("button", { name: /add model limit/i }));

    expect(screen.getByPlaceholderText("TPM Limit")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Input TPM Limit")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Output TPM Limit")).toBeInTheDocument();
    expect(screen.getByLabelText("TPM Limit")).toBeInTheDocument();
    expect(screen.getByLabelText("Input TPM Limit")).toBeInTheDocument();
    expect(screen.getByLabelText("Output TPM Limit")).toBeInTheDocument();
  });
});
