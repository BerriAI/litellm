import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import { ProjectsPage } from "./ProjectsPage";
import { ProjectResponse } from "@/app/(dashboard)/hooks/projects/useProjects";

const mockUseProjects = vi.fn();
vi.mock("@/app/(dashboard)/hooks/projects/useProjects", () => ({
  useProjects: () => mockUseProjects(),
}));

const mockUseTeams = vi.fn();
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeams: () => mockUseTeams(),
}));

// Stub modals and the detail page to keep tests focused on the list page
vi.mock("./ProjectModals/CreateProjectModal", () => ({
  CreateProjectModal: ({ isOpen }: { isOpen: boolean }) =>
    isOpen ? <div data-testid="create-modal" /> : null,
}));

vi.mock("./ProjectDetailsPage", () => ({
  ProjectDetail: ({ projectId }: { projectId: string }) => (
    <div data-testid="project-detail">{projectId}</div>
  ),
}));

const mockProjects: ProjectResponse[] = [
  {
    project_id: "proj-1",
    project_alias: "Alpha Project",
    description: "First project",
    team_id: "team-1",
    budget_id: null,
    metadata: null,
    models: ["gpt-4", "claude-3"],
    spend: 5.0,
    model_spend: null,
    model_rpm_limit: null,
    model_tpm_limit: null,
    blocked: false,
    object_permission_id: null,
    created_at: "2024-01-01T00:00:00Z",
    created_by: "user-1",
    updated_at: "2024-01-01T00:00:00Z",
    updated_by: "user-1",
    litellm_budget_table: null,
  },
  {
    project_id: "proj-2",
    project_alias: "Beta Project",
    description: "Second project",
    team_id: "team-2",
    budget_id: null,
    metadata: null,
    models: [],
    spend: 0,
    model_spend: null,
    model_rpm_limit: null,
    model_tpm_limit: null,
    blocked: true,
    object_permission_id: null,
    created_at: "2024-02-01T00:00:00Z",
    created_by: "user-2",
    updated_at: "2024-02-01T00:00:00Z",
    updated_by: "user-2",
    litellm_budget_table: null,
  },
];

describe("ProjectsPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseTeams.mockReturnValue({ data: [], isLoading: false });
  });

  it("should render the Projects heading", () => {
    mockUseProjects.mockReturnValue({ data: [], isLoading: false });
    renderWithProviders(<ProjectsPage />);
    expect(screen.getByRole("heading", { name: /projects/i })).toBeInTheDocument();
  });

  it("should show a 'Create Project' button", () => {
    mockUseProjects.mockReturnValue({ data: [], isLoading: false });
    renderWithProviders(<ProjectsPage />);
    expect(screen.getByRole("button", { name: /create project/i })).toBeInTheDocument();
  });

  it("should render the projects table", () => {
    mockUseProjects.mockReturnValue({ data: mockProjects, isLoading: false });
    renderWithProviders(<ProjectsPage />);
    expect(screen.getByText("Alpha Project")).toBeInTheDocument();
    expect(screen.getByText("Beta Project")).toBeInTheDocument();
  });

  it("should show the model count for each project", () => {
    mockUseProjects.mockReturnValue({ data: mockProjects, isLoading: false });
    renderWithProviders(<ProjectsPage />);
    // proj-1 has 2 models, proj-2 has 0
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("0")).toBeInTheDocument();
  });

  it("should display 'Active' tag for non-blocked projects", () => {
    mockUseProjects.mockReturnValue({ data: [mockProjects[0]], isLoading: false });
    renderWithProviders(<ProjectsPage />);
    expect(screen.getByText("Active")).toBeInTheDocument();
  });

  it("should display 'Blocked' tag for blocked projects", () => {
    mockUseProjects.mockReturnValue({ data: [mockProjects[1]], isLoading: false });
    renderWithProviders(<ProjectsPage />);
    expect(screen.getByText("Blocked")).toBeInTheDocument();
  });

  it("should open the create modal when 'Create Project' is clicked", async () => {
    const user = userEvent.setup();
    mockUseProjects.mockReturnValue({ data: [], isLoading: false });
    renderWithProviders(<ProjectsPage />);
    await user.click(screen.getByRole("button", { name: /create project/i }));
    expect(screen.getByTestId("create-modal")).toBeInTheDocument();
  });

  it("should show the project detail view when a project ID is clicked", async () => {
    const user = userEvent.setup();
    mockUseProjects.mockReturnValue({ data: mockProjects, isLoading: false });
    renderWithProviders(<ProjectsPage />);
    await user.click(screen.getByText("proj-1"));
    expect(screen.getByTestId("project-detail")).toHaveTextContent("proj-1");
  });

  it("should filter displayed projects when the search input has a value", async () => {
    const user = userEvent.setup();
    mockUseProjects.mockReturnValue({ data: mockProjects, isLoading: false });
    renderWithProviders(<ProjectsPage />);
    await user.type(
      screen.getByPlaceholderText(/search projects/i),
      "Alpha"
    );
    await waitFor(() => {
      expect(screen.getByText("Alpha Project")).toBeInTheDocument();
      expect(screen.queryByText("Beta Project")).not.toBeInTheDocument();
    });
  });

  it("should show the total project count in the pagination", () => {
    mockUseProjects.mockReturnValue({ data: mockProjects, isLoading: false });
    renderWithProviders(<ProjectsPage />);
    expect(screen.getByText("2 projects")).toBeInTheDocument();
  });

  it("should resolve team alias from the teams list in the Team column", () => {
    mockUseTeams.mockReturnValue({
      data: [{ team_id: "team-1", team_alias: "Engineering", models: [] }],
      isLoading: false,
    });
    mockUseProjects.mockReturnValue({ data: [mockProjects[0]], isLoading: false });
    renderWithProviders(<ProjectsPage />);
    expect(screen.getByText("Engineering")).toBeInTheDocument();
  });
});
