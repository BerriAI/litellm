import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen } from "../../../../tests/test-utils";
import { CreateProjectModal } from "./CreateProjectModal";

const mockMutate = vi.fn();
vi.mock("@/app/(dashboard)/hooks/projects/useCreateProject", () => ({
  useCreateProject: () => ({ mutate: mockMutate, isPending: false }),
}));

// Mock the form to keep tests focused on modal behavior
vi.mock("./ProjectBaseForm", () => ({
  ProjectBaseForm: () => <div data-testid="project-base-form" />,
}));

describe("CreateProjectModal", () => {
  const onClose = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should not render modal content when closed", () => {
    renderWithProviders(<CreateProjectModal isOpen={false} onClose={onClose} />);
    expect(screen.queryByText("Create New Project")).not.toBeInTheDocument();
  });

  it("should render the modal when open", () => {
    renderWithProviders(<CreateProjectModal isOpen={true} onClose={onClose} />);
    expect(screen.getByText("Create New Project")).toBeInTheDocument();
  });

  it("should show a 'Create Project' submit button", () => {
    renderWithProviders(<CreateProjectModal isOpen={true} onClose={onClose} />);
    expect(screen.getByRole("button", { name: /create project/i })).toBeInTheDocument();
  });

  it("should show a 'Cancel' button", () => {
    renderWithProviders(<CreateProjectModal isOpen={true} onClose={onClose} />);
    expect(screen.getByRole("button", { name: /cancel/i })).toBeInTheDocument();
  });

  it("should call onClose when the Cancel button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CreateProjectModal isOpen={true} onClose={onClose} />);
    await user.click(screen.getByRole("button", { name: /cancel/i }));
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("should render the project form inside the modal", () => {
    renderWithProviders(<CreateProjectModal isOpen={true} onClose={onClose} />);
    expect(screen.getByTestId("project-base-form")).toBeInTheDocument();
  });
});
