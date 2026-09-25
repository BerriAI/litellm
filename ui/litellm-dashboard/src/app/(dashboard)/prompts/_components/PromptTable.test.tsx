import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/../tests/test-utils";
import { PromptSpec } from "@/components/networking";

import PromptTable from "./PromptTable";

vi.mock("@/components/networking", () => ({
  modelHubCall: vi.fn().mockResolvedValue({ data: [] }),
}));

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));

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

const manyPrompts = (count: number): PromptSpec[] =>
  Array.from({ length: count }, (_, index) => ({
    prompt_id: `prompt-${String(index).padStart(2, "0")}`,
    litellm_params: { prompt_id: `prompt-${index}` },
    prompt_info: { prompt_type: "dotprompt" },
    created_at: new Date(Date.UTC(2025, 0, 1 + index)).toISOString(),
    updated_at: new Date(Date.UTC(2025, 0, 1 + index)).toISOString(),
  }));

const crossOrderedPrompts: PromptSpec[] = [mockPrompts[0], { ...mockPrompts[1], updated_at: "2026-01-01T00:00:00Z" }];

const firstRowPromptId = () =>
  within(screen.getAllByRole("row")[1]).getByRole("button", { name: /^prompt-/ }).textContent;

describe("PromptTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render every column header", () => {
    renderWithProviders(<PromptTable {...defaultProps} />);
    for (const header of ["Prompt ID", "Model", "Created At", "Updated At", "Environment", "Created By", "Type"]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
  });

  it("links the Created By cell to the creator's detail page, leaving the placeholder unlinked", () => {
    const prompts = [mockPrompts[0], { ...mockPrompts[1], created_by: "default_user_id" }];
    renderWithProviders(<PromptTable {...defaultProps} promptsList={prompts} />);

    expect(screen.getByRole("link", { name: "user-1" })).toHaveAttribute("href", "/ui/users?user=user-1");
    expect(screen.getByText("default_user_id")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "default_user_id" })).not.toBeInTheDocument();
  });

  it("should display the empty state when data is empty", () => {
    renderWithProviders(<PromptTable {...defaultProps} promptsList={[]} />);
    expect(screen.getByText("No prompts yet")).toBeInTheDocument();
  });

  it("should sort by created date descending by default", () => {
    renderWithProviders(<PromptTable {...defaultProps} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("prompt-newer")).toBeInTheDocument();
    expect(within(rows[1]).getByText("prompt-older")).toBeInTheDocument();
  });

  it("should call onPromptClick with the row's environment, defaulting to development", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PromptTable {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: "prompt-newer" }));
    expect(mockOnPromptClick).toHaveBeenCalledWith("prompt-newer", "production");
    await user.click(screen.getByRole("button", { name: "prompt-older" }));
    expect(mockOnPromptClick).toHaveBeenCalledWith("prompt-older", "development");
  });

  it("should label the environment and default missing environments to development", () => {
    renderWithProviders(<PromptTable {...defaultProps} />);
    expect(screen.getByText("production")).toBeInTheDocument();
    expect(screen.getByText("development")).toBeInTheDocument();
  });

  it("should delete a prompt through the actions menu when admin", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PromptTable {...defaultProps} />);
    await user.click(screen.getByTestId("prompt-actions-prompt-newer"));
    await user.click(await screen.findByTestId("prompt-action-delete"));
    expect(mockOnDeleteClick).toHaveBeenCalledWith("prompt-newer", "prompt-newer", "production");
  });

  it("should copy the prompt ID through the actions menu", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PromptTable {...defaultProps} />);
    await user.click(screen.getByTestId("prompt-actions-prompt-newer"));
    await user.click(await screen.findByTestId("prompt-action-copy"));
    expect(await window.navigator.clipboard.readText()).toBe("prompt-newer");
  });

  it("should hide the delete action for non-admins but keep copy available", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PromptTable {...defaultProps} isAdmin={false} />);
    await user.click(screen.getByTestId("prompt-actions-prompt-newer"));
    expect(await screen.findByTestId("prompt-action-copy")).toBeInTheDocument();
    expect(screen.queryByTestId("prompt-action-delete")).not.toBeInTheDocument();
  });

  describe("URL state", () => {
    it.each([
      ["prompt_id", "prompt-newer"],
      ["created_at", "prompt-older"],
      ["updated_at", "prompt-newer"],
    ])("orders rows ascending by %s from the URL", (sortBy, expectedFirstId) => {
      renderWithProviders(<PromptTable {...defaultProps} promptsList={crossOrderedPrompts} />, {
        searchParams: { sort_by: sortBy, sort_order: "asc" },
      });
      expect(firstRowPromptId()).toBe(expectedFirstId);
    });

    it("orders rows descending by prompt_id from the URL", () => {
      renderWithProviders(<PromptTable {...defaultProps} />, {
        searchParams: { sort_by: "prompt_id", sort_order: "desc" },
      });
      expect(firstRowPromptId()).toBe("prompt-older");
    });

    it("opens on the page in the URL", () => {
      renderWithProviders(<PromptTable {...defaultProps} promptsList={manyPrompts(27)} />, {
        searchParams: { page: "2" },
      });
      expect(screen.getAllByRole("row")).toHaveLength(3);
      expect(firstRowPromptId()).toBe("prompt-01");
    });

    it("writes the clicked sort column to the URL", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<PromptTable {...defaultProps} />, { onUrlUpdate });

      await user.click(screen.getByTestId("sort-header-prompt_id"));

      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      const params = onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;
      expect(params?.get("sort_by")).toBe("prompt_id");
      expect(params?.get("sort_order")).toBe("asc");
      expect(firstRowPromptId()).toBe("prompt-newer");
    });

    it("writes the next page to the URL", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<PromptTable {...defaultProps} promptsList={manyPrompts(27)} />, { onUrlUpdate });
      expect(firstRowPromptId()).toBe("prompt-26");

      await user.click(screen.getByRole("button", { name: "Go to next page" }));

      await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("page")).toBe("2"));
      expect(firstRowPromptId()).toBe("prompt-01");
    });
  });
});
