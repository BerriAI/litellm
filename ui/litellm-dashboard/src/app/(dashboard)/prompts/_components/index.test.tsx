import { screen, waitFor } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { deletePromptCall, getPromptsList } from "@/components/networking";

import PromptsPanel from "./index";
import { chooseSelectOption, renderWithProviders } from "../../../../../tests/test-utils";

vi.mock("@/components/networking", () => ({
  getPromptsList: vi.fn(),
  deletePromptCall: vi.fn(),
}));

vi.mock("./PromptTable", () => ({
  __esModule: true,
  default: ({
    isLoading,
    onPromptClick,
    onDeleteClick,
  }: {
    isLoading: boolean;
    onPromptClick: (id: string, environment: string) => void;
    onDeleteClick: (id: string, name: string, environment: string) => void;
  }) => (
    <div data-testid="prompt-table">
      {isLoading ? "table-loading" : "table-loaded"}
      <button type="button" onClick={() => onPromptClick("prompt-1", "staging")}>
        row-open
      </button>
      <button type="button" onClick={() => onDeleteClick("prompt-1", "my-prompt", "staging")}>
        row-delete
      </button>
    </div>
  ),
}));

vi.mock("./prompt_info", () => ({
  __esModule: true,
  default: ({
    initialEnvironment,
    onClose,
    onEdit,
  }: {
    initialEnvironment?: string;
    onClose: () => void;
    onEdit: (promptData: { prompt_id: string }) => void;
  }) => (
    <div>
      <div>prompt-info-view:{initialEnvironment ?? "none"}</div>
      <button type="button" onClick={onClose}>
        info-close
      </button>
      <button type="button" onClick={() => onEdit({ prompt_id: "prompt-1" })}>
        info-edit
      </button>
    </div>
  ),
}));
vi.mock("./add_prompt_form", () => ({
  __esModule: true,
  default: ({ visible }: { visible: boolean }) => (visible ? <div>add-prompt-form</div> : null),
}));
vi.mock("./prompt_editor_view", () => ({
  __esModule: true,
  default: ({
    initialPromptData,
    onClose,
  }: {
    initialPromptData: { prompt_id: string } | null;
    onClose: () => void;
  }) => (
    <div>
      <div>prompt-editor-view</div>
      <div>editor-target:{initialPromptData?.prompt_id ?? "new"}</div>
      <button type="button" onClick={onClose}>
        editor-close
      </button>
    </div>
  ),
}));

const mockGetPromptsList = vi.mocked(getPromptsList);
const mockDeletePromptCall = vi.mocked(deletePromptCall);

interface RenderPanelOptions {
  searchParams?: Record<string, string>;
  onUrlUpdate?: OnUrlUpdateFunction;
}

const renderPanel = (userRole?: string, options: RenderPanelOptions = {}) =>
  renderWithProviders(<PromptsPanel accessToken="sk-test" userRole={userRole ?? "Admin"} />, options);

const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) => onUrlUpdate.mock.calls.at(-1)?.[0];

describe("PromptsPanel loading state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetPromptsList.mockResolvedValue({ prompts: [] } as never);
  });

  it("should resolve the loading state when accessToken is null instead of showing the skeleton forever", async () => {
    renderWithProviders(<PromptsPanel accessToken={null} />);
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
    renderWithProviders(<PromptsPanel accessToken="sk-test" userRole="Admin" />);
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

    await chooseSelectOption(user, screen.getByRole("combobox"), "Production");

    await waitFor(() => expect(mockGetPromptsList).toHaveBeenLastCalledWith("sk-test", "production"));
  });

  it("should show the picked environment by label and clear back to the unfiltered list", async () => {
    // Base UI's exit animation never completes in jsdom, so the closing popup keeps
    // pointer-events: none and blocks the second open. The clicks still dispatch.
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderPanel("Admin");
    await screen.findByText("table-loaded");

    await chooseSelectOption(user, screen.getByRole("combobox"), "Production");
    await waitFor(() => expect(screen.getByRole("combobox")).toHaveTextContent("Production"));

    await chooseSelectOption(user, screen.getByRole("combobox"), "All Environments");

    await waitFor(() => expect(screen.getByRole("combobox")).toHaveTextContent("All Environments"));
    await waitFor(() => expect(mockGetPromptsList).toHaveBeenLastCalledWith("sk-test", undefined));
  });
});

describe("PromptsPanel row navigation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetPromptsList.mockResolvedValue({ prompts: [] } as never);
  });

  it("should open the info view preselected to the clicked row's environment", async () => {
    const user = userEvent.setup();
    renderPanel("Admin");

    await user.click(await screen.findByRole("button", { name: "row-open" }));

    expect(screen.getByText("prompt-info-view:staging")).toBeInTheDocument();
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

    expect(await screen.findByText(/the staging copy of prompt: my-prompt/i)).toBeInTheDocument();
    expect(screen.getByText(/cannot be undone/i)).toBeInTheDocument();
    expect(mockDeletePromptCall).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: /^delete$/i }));

    await waitFor(() => expect(mockDeletePromptCall).toHaveBeenCalledWith("sk-test", "prompt-1", "staging"));
  });

  it("should abandon the delete when the confirmation is dismissed", async () => {
    const user = userEvent.setup();
    renderPanel("Admin");

    await user.click(await screen.findByRole("button", { name: "row-delete" }));
    await screen.findByText(/the staging copy of prompt: my-prompt/i);

    await user.click(screen.getByRole("button", { name: /cancel/i }));

    await waitFor(() => expect(screen.queryByText(/the staging copy of prompt: my-prompt/i)).not.toBeInTheDocument());
    expect(mockDeletePromptCall).not.toHaveBeenCalled();
  });

  it("should keep the confirmation up while the delete request is still in flight", async () => {
    const user = userEvent.setup();
    let finishDelete: () => void = () => {};
    mockDeletePromptCall.mockReturnValue(
      new Promise<void>((resolve) => {
        finishDelete = () => resolve();
      }) as never,
    );
    renderPanel("Admin");

    await user.click(await screen.findByRole("button", { name: "row-delete" }));
    await screen.findByText(/the staging copy of prompt: my-prompt/i);
    await user.click(screen.getByRole("button", { name: /^delete$/i }));
    await waitFor(() => expect(mockDeletePromptCall).toHaveBeenCalledWith("sk-test", "prompt-1", "staging"));

    await user.keyboard("{Escape}");
    expect(screen.getByText(/the staging copy of prompt: my-prompt/i)).toBeInTheDocument();

    finishDelete();
    await waitFor(() => expect(screen.queryByText(/the staging copy of prompt: my-prompt/i)).not.toBeInTheDocument());
  });
});

describe("PromptsPanel URL state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetPromptsList.mockResolvedValue({ prompts: [] } as never);
  });

  it("scopes the list to the env in the URL", async () => {
    renderPanel("Admin", { searchParams: { env: "production" } });

    await waitFor(() => expect(mockGetPromptsList).toHaveBeenCalledWith("sk-test", "production"));
    expect(mockGetPromptsList).not.toHaveBeenCalledWith("sk-test", undefined);
    expect(screen.getByRole("combobox")).toHaveTextContent("Production");
  });

  it("treats an unknown env in the URL as all environments", async () => {
    renderPanel("Admin", { searchParams: { env: "qa" } });

    await waitFor(() => expect(mockGetPromptsList).toHaveBeenCalledWith("sk-test", undefined));
    expect(screen.getByRole("combobox")).toHaveTextContent("All Environments");
  });

  it("writes the picked env to the URL and drops it again for all environments", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderPanel("Admin", { onUrlUpdate });
    await screen.findByText("table-loaded");

    await chooseSelectOption(user, screen.getByRole("combobox"), "Production");
    await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("env")).toBe("production"));

    await chooseSelectOption(user, screen.getByRole("combobox"), "All Environments");
    await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.has("env")).toBe(false));
  });

  it("opens the prompt and environment named in the URL", async () => {
    renderPanel("Admin", { searchParams: { prompt: "prompt-1", prompt_env: "staging" } });

    expect(await screen.findByText("prompt-info-view:staging")).toBeInTheDocument();
    expect(screen.queryByTestId("prompt-table")).not.toBeInTheDocument();
  });

  it("pushes the opened prompt to the URL and clears it on close", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderPanel("Admin", { searchParams: { version: "3", tab: "raw" }, onUrlUpdate });

    await user.click(await screen.findByRole("button", { name: "row-open" }));

    await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("prompt")).toBe("prompt-1"));
    expect(lastUrl(onUrlUpdate)?.searchParams.get("prompt_env")).toBe("staging");
    expect(lastUrl(onUrlUpdate)?.searchParams.has("version")).toBe(false);
    expect(lastUrl(onUrlUpdate)?.searchParams.has("tab")).toBe(false);
    expect(lastUrl(onUrlUpdate)?.options.history).toBe("push");

    await user.click(screen.getByRole("button", { name: "info-close" }));

    await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.has("prompt")).toBe(false));
    expect(lastUrl(onUrlUpdate)?.searchParams.has("prompt_env")).toBe(false);
    expect(await screen.findByTestId("prompt-table")).toBeInTheDocument();
  });

  it("opens the new prompt editor from view=editor for an admin", async () => {
    renderPanel("Admin", { searchParams: { view: "editor" } });

    expect(await screen.findByText("editor-target:new")).toBeInTheDocument();
    expect(screen.queryByTestId("prompt-table")).not.toBeInTheDocument();
  });

  it("keeps a read-only viewer on the table despite view=editor", async () => {
    renderPanel("Admin Viewer", { searchParams: { view: "editor" } });

    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(screen.queryByText("prompt-editor-view")).not.toBeInTheDocument();
  });

  it("pushes view=editor when the add action is used and drops it when the editor closes", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderPanel("Admin", { onUrlUpdate });

    await user.click(await screen.findByRole("button", { name: /add new prompt/i }));

    await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("view")).toBe("editor"));
    expect(lastUrl(onUrlUpdate)?.options.history).toBe("push");

    await user.click(screen.getByRole("button", { name: "editor-close" }));

    await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.has("view")).toBe(false));
    expect(await screen.findByTestId("prompt-table")).toBeInTheDocument();
  });

  it("shows the prompt instead of an empty editor when view=editor points at a prompt with no edit session", async () => {
    renderPanel("Admin", { searchParams: { view: "editor", prompt: "prompt-1", prompt_env: "staging" } });

    expect(await screen.findByText("prompt-info-view:staging")).toBeInTheDocument();
    expect(screen.queryByText("prompt-editor-view")).not.toBeInTheDocument();
  });

  it("edits the open prompt under view=editor and returns to it when the editor closes", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderPanel("Admin", { searchParams: { prompt: "prompt-1", prompt_env: "staging" }, onUrlUpdate });

    await user.click(await screen.findByRole("button", { name: "info-edit" }));

    expect(await screen.findByText("editor-target:prompt-1")).toBeInTheDocument();
    await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("view")).toBe("editor"));
    expect(lastUrl(onUrlUpdate)?.searchParams.get("prompt")).toBe("prompt-1");
    expect(lastUrl(onUrlUpdate)?.options.history).toBe("push");

    await user.click(screen.getByRole("button", { name: "editor-close" }));

    expect(await screen.findByText("prompt-info-view:staging")).toBeInTheDocument();
    await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.has("view")).toBe(false));
    expect(lastUrl(onUrlUpdate)?.searchParams.get("prompt")).toBe("prompt-1");
  });
});
