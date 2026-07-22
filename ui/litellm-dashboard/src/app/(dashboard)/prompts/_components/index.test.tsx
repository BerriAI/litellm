import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { deletePromptCall, getPromptsList } from "@/components/networking";

import PromptsPanel from "./index";

vi.mock("@/components/networking", () => ({
  getPromptsList: vi.fn(),
  deletePromptCall: vi.fn(),
}));

vi.mock("./PromptTable", () => ({
  __esModule: true,
  default: ({
    isLoading,
    onDeleteClick,
  }: {
    isLoading: boolean;
    onDeleteClick: (id: string, name: string) => void;
  }) => (
    <div data-testid="prompt-table">
      {isLoading ? "table-loading" : "table-loaded"}
      <button type="button" onClick={() => onDeleteClick("prompt-1", "my-prompt")}>
        row-delete
      </button>
    </div>
  ),
}));

vi.mock("./prompt_info", () => ({ __esModule: true, default: () => <div>prompt-info-view</div> }));
vi.mock("./add_prompt_form", () => ({
  __esModule: true,
  default: ({ visible }: { visible: boolean }) => (visible ? <div>add-prompt-form</div> : null),
}));
vi.mock("./prompt_editor_view", () => ({ __esModule: true, default: () => <div>prompt-editor-view</div> }));

const mockGetPromptsList = vi.mocked(getPromptsList);
const mockDeletePromptCall = vi.mocked(deletePromptCall);

const renderPanel = (userRole?: string) =>
  render(<PromptsPanel accessToken="sk-test" userRole={userRole ?? "Admin"} />);

describe("PromptsPanel loading state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetPromptsList.mockResolvedValue({ prompts: [] } as never);
  });

  it("should resolve the loading state when accessToken is null instead of showing the skeleton forever", async () => {
    render(<PromptsPanel accessToken={null} />);
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockGetPromptsList).not.toHaveBeenCalled();
  });

  it("should show the loading state until the prompt fetch settles", async () => {
    let resolveFetch: (value: { prompts: never[] }) => void = () => {};
    mockGetPromptsList.mockReturnValue(
      new Promise((resolve) => {
        resolveFetch = resolve;
      }) as never,
    );
    render(<PromptsPanel accessToken="sk-test" userRole="Admin" />);
    expect(screen.getByText("table-loading")).toBeInTheDocument();

    resolveFetch({ prompts: [] });
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockGetPromptsList).toHaveBeenCalledWith("sk-test", undefined);
  });
});

describe("PromptsPanel toolbar", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetPromptsList.mockResolvedValue({ prompts: [] } as never);
  });

  it("should offer both create actions to a proxy admin", async () => {
    renderPanel("Admin");

    expect(await screen.findByRole("button", { name: /add new prompt/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /upload \.prompt file/i })).toBeEnabled();
  });

  it("should hide both create actions from a read-only viewer", async () => {
    renderPanel("Admin Viewer");

    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /add new prompt/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /upload \.prompt file/i })).not.toBeInTheDocument();
  });

  it("should open the editor view when the add action is used", async () => {
    const user = userEvent.setup();
    renderPanel("Admin");

    await user.click(await screen.findByRole("button", { name: /add new prompt/i }));

    expect(screen.getByText("prompt-editor-view")).toBeInTheDocument();
    expect(screen.queryByTestId("prompt-table")).not.toBeInTheDocument();
  });

  it("should open the upload form when the upload action is used", async () => {
    const user = userEvent.setup();
    renderPanel("Admin");

    expect(screen.queryByText("add-prompt-form")).not.toBeInTheDocument();
    await user.click(await screen.findByRole("button", { name: /upload \.prompt file/i }));

    expect(screen.getByText("add-prompt-form")).toBeInTheDocument();
  });

  it("should refetch scoped to the environment picked in the filter", async () => {
    const user = userEvent.setup();
    renderPanel("Admin");
    await screen.findByText("table-loaded");

    expect(screen.getByText("All Environments")).toBeInTheDocument();

    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByText("Production"));

    await waitFor(() => expect(mockGetPromptsList).toHaveBeenLastCalledWith("sk-test", "production"));
  });
});

describe("PromptsPanel delete confirmation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetPromptsList.mockResolvedValue({ prompts: [] } as never);
    mockDeletePromptCall.mockResolvedValue(undefined as never);
  });

  it("should not delete until the confirmation is accepted", async () => {
    const user = userEvent.setup();
    renderPanel("Admin");

    await user.click(await screen.findByRole("button", { name: "row-delete" }));

    expect(await screen.findByText(/delete prompt: my-prompt/i)).toBeInTheDocument();
    expect(screen.getByText(/cannot be undone/i)).toBeInTheDocument();
    expect(mockDeletePromptCall).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: /^delete$/i }));

    await waitFor(() => expect(mockDeletePromptCall).toHaveBeenCalledWith("sk-test", "prompt-1"));
  });

  it("should abandon the delete when the confirmation is dismissed", async () => {
    const user = userEvent.setup();
    renderPanel("Admin");

    await user.click(await screen.findByRole("button", { name: "row-delete" }));
    await screen.findByText(/delete prompt: my-prompt/i);

    await user.click(screen.getByRole("button", { name: /cancel/i }));

    await waitFor(() => expect(screen.queryByText(/delete prompt: my-prompt/i)).not.toBeInTheDocument());
    expect(mockDeletePromptCall).not.toHaveBeenCalled();
  });
});
