import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PromptSpec } from "@/components/networking";

import PromptTable from "./PromptTable";

vi.mock("@/components/networking", () => ({
  modelHubCall: vi.fn().mockResolvedValue({ data: [] }),
}));

const mockPrompts: PromptSpec[] = [
  {
    prompt_id: "prompt-newer",
    litellm_params: { prompt_id: "prompt-newer" },
    prompt_info: { prompt_type: "dotprompt" },
    created_at: "2025-01-15T10:30:00Z",
    updated_at: "2025-01-15T11:00:00Z",
    environment: "production",
    created_by: "user-1",
  },
  {
    prompt_id: "prompt-older",
    litellm_params: { prompt_id: "prompt-older" },
    prompt_info: { prompt_type: "dotprompt" },
    created_at: "2024-01-10T09:15:00Z",
    updated_at: "2024-01-12T14:20:00Z",
  },
];

const mockOnPromptClick = vi.fn();
const mockOnDeleteClick = vi.fn();

const defaultProps = {
  promptsList: mockPrompts,
  isLoading: false,
  onPromptClick: mockOnPromptClick,
  onDeleteClick: mockOnDeleteClick,
  accessToken: null,
  isAdmin: true,
};

describe("PromptTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render every column header", () => {
    render(<PromptTable {...defaultProps} />);
    for (const header of ["Prompt ID", "Model", "Created At", "Updated At", "Environment", "Created By", "Type"]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
  });

  it("should display the empty state when data is empty", () => {
    render(<PromptTable {...defaultProps} promptsList={[]} />);
    expect(screen.getByText("No prompts yet")).toBeInTheDocument();
  });

  it("should sort by created date descending by default", () => {
    render(<PromptTable {...defaultProps} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("prompt-newer")).toBeInTheDocument();
    expect(within(rows[1]).getByText("prompt-older")).toBeInTheDocument();
  });

  it("should call onPromptClick when the prompt ID is clicked", async () => {
    const user = userEvent.setup();
    render(<PromptTable {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: "prompt-newer" }));
    expect(mockOnPromptClick).toHaveBeenCalledWith("prompt-newer");
  });

  it("should label the environment and default missing environments to development", () => {
    render(<PromptTable {...defaultProps} />);
    expect(screen.getByText("production")).toBeInTheDocument();
    expect(screen.getByText("development")).toBeInTheDocument();
  });

  it("should delete a prompt through the actions menu when admin", async () => {
    const user = userEvent.setup();
    render(<PromptTable {...defaultProps} />);
    await user.click(screen.getByTestId("prompt-actions-prompt-newer"));
    await user.click(await screen.findByTestId("prompt-action-delete"));
    expect(mockOnDeleteClick).toHaveBeenCalledWith("prompt-newer", "prompt-newer");
  });

  it("should copy the prompt ID through the actions menu", async () => {
    const user = userEvent.setup();
    render(<PromptTable {...defaultProps} />);
    await user.click(screen.getByTestId("prompt-actions-prompt-newer"));
    await user.click(await screen.findByTestId("prompt-action-copy"));
    expect(await window.navigator.clipboard.readText()).toBe("prompt-newer");
  });

  it("should hide the delete action for non-admins but keep copy available", async () => {
    const user = userEvent.setup();
    render(<PromptTable {...defaultProps} isAdmin={false} />);
    await user.click(screen.getByTestId("prompt-actions-prompt-newer"));
    expect(await screen.findByTestId("prompt-action-copy")).toBeInTheDocument();
    expect(screen.queryByTestId("prompt-action-delete")).not.toBeInTheDocument();
  });
});
