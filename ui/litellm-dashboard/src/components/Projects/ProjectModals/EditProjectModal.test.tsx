import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen } from "../../../../tests/test-utils";
import { EditProjectModal } from "./EditProjectModal";
import { ProjectResponse } from "@/app/(dashboard)/hooks/projects/useProjects";

const mockMutate = vi.fn();
vi.mock("@/app/(dashboard)/hooks/projects/useUpdateProject", () => ({
  useUpdateProject: () => ({ mutate: mockMutate, isPending: false }),
}));

vi.mock("./ProjectBaseForm", () => ({
  ProjectBaseForm: () => <div data-testid="project-base-form" />,
}));

const mockProject: ProjectResponse = {
  project_id: "proj-1",
  project_alias: "My Project",
  description: "A test project",
  team_id: "team-1",
  budget_id: null,
  metadata: null,
  models: ["gpt-4"],
  spend: 10.0,
  model_spend: null,
  model_rpm_limit: null,
  model_tpm_limit: null,
  blocked: false,
  object_permission_id: null,
  created_at: "2024-01-01T00:00:00Z",
  created_by: "user-1",
  updated_at: "2024-01-02T00:00:00Z",
  updated_by: "user-1",
  litellm_budget_table: null,
};

describe("EditProjectModal", () => {
  const onClose = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should not render modal content when closed", () => {
    renderWithProviders(
      <EditProjectModal isOpen={false} project={mockProject} onClose={onClose} />
    );
    expect(screen.queryByText("Edit Project")).not.toBeInTheDocument();
  });

  it("should render the modal when open", () => {
    renderWithProviders(
      <EditProjectModal isOpen={true} project={mockProject} onClose={onClose} />
    );
    expect(screen.getByText("Edit Project")).toBeInTheDocument();
  });

  it("should show a 'Save Changes' submit button", () => {
    renderWithProviders(
      <EditProjectModal isOpen={true} project={mockProject} onClose={onClose} />
    );
    expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
  });

  it("should show a 'Cancel' button", () => {
    renderWithProviders(
      <EditProjectModal isOpen={true} project={mockProject} onClose={onClose} />
    );
    expect(screen.getByRole("button", { name: /cancel/i })).toBeInTheDocument();
  });

  it("should call onClose when the Cancel button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <EditProjectModal isOpen={true} project={mockProject} onClose={onClose} />
    );
    await user.click(screen.getByRole("button", { name: /cancel/i }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("should render the project form inside the modal", () => {
    renderWithProviders(
      <EditProjectModal isOpen={true} project={mockProject} onClose={onClose} />
    );
    expect(screen.getByTestId("project-base-form")).toBeInTheDocument();
  });
});
