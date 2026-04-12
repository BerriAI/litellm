import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen } from "../../../tests/test-utils";
import { ProjectDetail } from "./ProjectDetailsPage";
import { ProjectResponse } from "@/app/(dashboard)/hooks/projects/useProjects";

const mockUseProjectDetails = vi.fn();
vi.mock("@/app/(dashboard)/hooks/projects/useProjectDetails", () => ({
  useProjectDetails: (id: string) => mockUseProjectDetails(id),
}));

const mockUseTeam = vi.fn();
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeam: (id?: string) => mockUseTeam(id),
}));

vi.mock("./ProjectModals/EditProjectModal", () => ({
  EditProjectModal: ({ isOpen }: { isOpen: boolean }) =>
    isOpen ? <div data-testid="edit-modal" /> : null,
}));

vi.mock("@/components/common_components/DefaultProxyAdminTag", () => ({
  default: ({ userId }: { userId: string }) => <span>{userId}</span>,
}));

const mockProject: ProjectResponse = {
  project_id: "proj-1",
  project_alias: "My Project",
  description: "A sample project",
  team_id: "team-1",
  budget_id: null,
  metadata: null,
  models: ["gpt-4"],
  spend: 12.5,
  model_spend: { "gpt-4": 12.5 },
  model_rpm_limit: null,
  model_tpm_limit: null,
  blocked: false,
  object_permission_id: null,
  created_at: "2024-01-15T08:00:00Z",
  created_by: "user-1",
  updated_at: "2024-02-01T12:00:00Z",
  updated_by: "user-2",
  litellm_budget_table: null,
};

describe("ProjectDetail", () => {
  const onBack = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseTeam.mockReturnValue({ data: undefined, isLoading: false });
  });

  describe("when loading", () => {
    it("should show a loading spinner", () => {
      mockUseProjectDetails.mockReturnValue({ data: undefined, isLoading: true });
      renderWithProviders(<ProjectDetail projectId="proj-1" onBack={onBack} />);
      expect(screen.getByRole("img", { hidden: true })).toBeInTheDocument();
    });
  });

  describe("when the project is not found", () => {
    it("should display 'Project not found'", () => {
      mockUseProjectDetails.mockReturnValue({ data: undefined, isLoading: false });
      renderWithProviders(<ProjectDetail projectId="proj-1" onBack={onBack} />);
      expect(screen.getByText("Project not found")).toBeInTheDocument();
    });

    it("should call onBack when the back button is clicked in the not-found state", async () => {
      const user = userEvent.setup();
      mockUseProjectDetails.mockReturnValue({ data: undefined, isLoading: false });
      renderWithProviders(<ProjectDetail projectId="proj-1" onBack={onBack} />);
      await user.click(screen.getByRole("button"));
      expect(onBack).toHaveBeenCalledOnce();
    });
  });

  describe("when the project loads successfully", () => {
    beforeEach(() => {
      mockUseProjectDetails.mockReturnValue({ data: mockProject, isLoading: false });
    });

    it("should render", () => {
      renderWithProviders(<ProjectDetail projectId="proj-1" onBack={onBack} />);
      expect(screen.getByText("My Project")).toBeInTheDocument();
    });

    it("should display the project alias as the page title", () => {
      renderWithProviders(<ProjectDetail projectId="proj-1" onBack={onBack} />);
      expect(screen.getByRole("heading", { name: "My Project" })).toBeInTheDocument();
    });

    it("should display 'Active' for a non-blocked project", () => {
      renderWithProviders(<ProjectDetail projectId="proj-1" onBack={onBack} />);
      expect(screen.getByText("Active")).toBeInTheDocument();
    });

    it("should display 'Blocked' for a blocked project", () => {
      mockUseProjectDetails.mockReturnValue({
        data: { ...mockProject, blocked: true },
        isLoading: false,
      });
      renderWithProviders(<ProjectDetail projectId="proj-1" onBack={onBack} />);
      expect(screen.getByText("Blocked")).toBeInTheDocument();
    });

    it("should call onBack when the back button is clicked", async () => {
      const user = userEvent.setup();
      renderWithProviders(<ProjectDetail projectId="proj-1" onBack={onBack} />);
      await user.click(screen.getByRole("button", { name: "" }));
      expect(onBack).toHaveBeenCalledOnce();
    });

    it("should show the current spend amount", () => {
      renderWithProviders(<ProjectDetail projectId="proj-1" onBack={onBack} />);
      expect(screen.getByText("$12.50")).toBeInTheDocument();
    });

    it("should show 'No budget limit' when no max budget is set", () => {
      renderWithProviders(<ProjectDetail projectId="proj-1" onBack={onBack} />);
      expect(screen.getByText("No budget limit")).toBeInTheDocument();
    });

    it("should show the budget limit when one is set", () => {
      mockUseProjectDetails.mockReturnValue({
        data: {
          ...mockProject,
          litellm_budget_table: { max_budget: 100 },
        },
        isLoading: false,
      });
      renderWithProviders(<ProjectDetail projectId="proj-1" onBack={onBack} />);
      expect(screen.getByText("of $100.00 budget")).toBeInTheDocument();
    });

    it("should show the project description", () => {
      renderWithProviders(<ProjectDetail projectId="proj-1" onBack={onBack} />);
      expect(screen.getByText("A sample project")).toBeInTheDocument();
    });

    it("should show an 'Edit Project' button", () => {
      renderWithProviders(<ProjectDetail projectId="proj-1" onBack={onBack} />);
      expect(screen.getByRole("button", { name: /edit project/i })).toBeInTheDocument();
    });

    it("should open the edit modal when 'Edit Project' is clicked", async () => {
      const user = userEvent.setup();
      renderWithProviders(<ProjectDetail projectId="proj-1" onBack={onBack} />);
      await user.click(screen.getByRole("button", { name: /edit project/i }));
      expect(screen.getByTestId("edit-modal")).toBeInTheDocument();
    });

    it("should show 'No team assigned' when the project has no team", () => {
      mockUseProjectDetails.mockReturnValue({
        data: { ...mockProject, team_id: null },
        isLoading: false,
      });
      renderWithProviders(<ProjectDetail projectId="proj-1" onBack={onBack} />);
      expect(screen.getByText("No team assigned")).toBeInTheDocument();
    });

    it("should show team information when team data is available", () => {
      mockUseTeam.mockReturnValue({
        data: {
          team_info: {
            team_id: "team-1",
            team_alias: "Engineering",
            models: ["gpt-4"],
            spend: 50,
            members_with_roles: [],
          },
        },
        isLoading: false,
      });
      renderWithProviders(<ProjectDetail projectId="proj-1" onBack={onBack} />);
      expect(screen.getByText("Engineering")).toBeInTheDocument();
    });
  });
});
