import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, waitFor } from "@/../tests/test-utils";
import { getPromptVersions, PromptSpec } from "@/components/networking";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import VersionHistorySidePanel from "./VersionHistorySidePanel";

vi.mock("@/components/networking", () => ({ getPromptVersions: vi.fn() }));

const versions = [
  {
    prompt_id: "welcome.v2",
    version: 2,
    created_at: "2024-01-15T10:30:00Z",
    litellm_params: {},
    prompt_info: { prompt_type: "db" },
  },
  {
    prompt_id: "welcome.v1",
    version: 1,
    created_at: "2024-01-10T09:00:00Z",
    litellm_params: {},
    prompt_info: { prompt_type: "config" },
  },
] satisfies PromptSpec[];

const props = {
  isOpen: true,
  onClose: vi.fn(),
  accessToken: "token",
  promptId: "welcome.v2",
  activeVersionId: "welcome.v2",
  onSelectVersion: vi.fn(),
};

describe("VersionHistorySidePanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getPromptVersions).mockResolvedValue({ prompts: versions });
  });

  it("should render prompt versions", async () => {
    renderWithProviders(<VersionHistorySidePanel {...props} />);

    expect(await screen.findByRole("dialog", { name: "Version History" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /v2/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /v1/i })).toBeInTheDocument();
  });

  it("should show an accessible loading state while versions are pending", () => {
    vi.mocked(getPromptVersions).mockImplementation(() => new Promise(() => {}));

    renderWithProviders(<VersionHistorySidePanel {...props} />);

    expect(screen.getByRole("status", { name: "Loading version history" })).toBeInTheDocument();
  });

  it("should show the empty state when no versions are available", async () => {
    vi.mocked(getPromptVersions).mockResolvedValue({ prompts: [] });

    renderWithProviders(<VersionHistorySidePanel {...props} />);

    expect(await screen.findByText("No version history available.")).toBeInTheDocument();
  });

  it("should display explicit version numbers", async () => {
    renderWithProviders(<VersionHistorySidePanel {...props} />);

    expect(await screen.findByText("v2")).toBeInTheDocument();
    expect(screen.getByText("v1")).toBeInTheDocument();
  });

  it.each([
    ["dot suffix", "welcome.v7", { prompt_id: "welcome.v7", litellm_params: {}, prompt_info: { prompt_type: "db" } }],
    [
      "underscore suffix",
      "welcome_v8",
      { prompt_id: "welcome_v8", litellm_params: {}, prompt_info: { prompt_type: "db" } },
    ],
    ["no suffix", "welcome", { prompt_id: "welcome", litellm_params: {}, prompt_info: { prompt_type: "db" } }],
  ])("should derive a version number from a %s", async (_case, promptId, prompt) => {
    vi.mocked(getPromptVersions).mockResolvedValue({ prompts: [prompt] });

    renderWithProviders(<VersionHistorySidePanel {...props} promptId={promptId} activeVersionId={undefined} />);

    const expectedVersions = new Map([
      ["welcome.v7", "v7"],
      ["welcome_v8", "v8"],
      ["welcome", "v1"],
    ]);
    const expectedVersion = expectedVersions.get(promptId)!;
    expect(await screen.findByText(expectedVersion)).toBeInTheDocument();
  });

  it("should label the first version as latest", async () => {
    renderWithProviders(<VersionHistorySidePanel {...props} />);

    expect(await screen.findByText("Latest")).toBeInTheDocument();
  });

  it.each(["welcome.v1", "welcome_v1"])("should mark %s as the active version", async (activeVersionId) => {
    renderWithProviders(<VersionHistorySidePanel {...props} activeVersionId={activeVersionId} />);

    const versionOne = await screen.findByRole("button", { name: /v1/i });
    expect(versionOne).toHaveTextContent("Active");
  });

  it("should mark the latest version active when no active version is supplied", async () => {
    renderWithProviders(<VersionHistorySidePanel {...props} activeVersionId={undefined} />);

    const latestVersion = await screen.findByRole("button", { name: /v2/i });
    expect(latestVersion).toHaveTextContent("Active");
  });

  it("should display creation dates", async () => {
    renderWithProviders(<VersionHistorySidePanel {...props} />);

    expect(await screen.findByText(new Date(versions[0].created_at).toLocaleString())).toBeInTheDocument();
  });

  it("should show a placeholder when a creation date is unavailable", async () => {
    vi.mocked(getPromptVersions).mockResolvedValue({
      prompts: [{ prompt_id: "welcome.v1", version: 1, litellm_params: {}, prompt_info: { prompt_type: "db" } }],
    });

    renderWithProviders(<VersionHistorySidePanel {...props} />);

    expect(await screen.findByText("-")).toBeInTheDocument();
  });

  it("should identify database and config prompt sources", async () => {
    renderWithProviders(<VersionHistorySidePanel {...props} />);

    expect(await screen.findByText("Saved to Database")).toBeInTheDocument();
    expect(screen.getByText("Config Prompt")).toBeInTheDocument();
  });

  it("should fetch versions using the base prompt identifier", async () => {
    renderWithProviders(<VersionHistorySidePanel {...props} promptId="welcome.v9" />);

    await waitFor(() => expect(getPromptVersions).toHaveBeenCalledWith("token", "welcome"));
  });

  it.each([
    ["closed", { isOpen: false }],
    ["without a token", { accessToken: null }],
    ["without a prompt identifier", { promptId: "" }],
  ])("should not fetch versions when %s", async (_case, override) => {
    renderWithProviders(<VersionHistorySidePanel {...props} {...override} />);

    await waitFor(() => expect(getPromptVersions).not.toHaveBeenCalled());
  });

  it("should refetch versions when the prompt identifier changes", async () => {
    const { rerender } = renderWithProviders(<VersionHistorySidePanel {...props} />);
    await waitFor(() => expect(getPromptVersions).toHaveBeenCalledWith("token", "welcome"));

    rerender(<VersionHistorySidePanel {...props} promptId="follow-up.v1" />);

    await waitFor(() => expect(getPromptVersions).toHaveBeenLastCalledWith("token", "follow-up"));
  });

  it("should show the empty state when loading versions fails", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    vi.mocked(getPromptVersions).mockRejectedValue(new Error("Network error"));

    renderWithProviders(<VersionHistorySidePanel {...props} />);

    expect(await screen.findByText("No version history available.")).toBeInTheDocument();
    expect(consoleError).toHaveBeenCalledWith("Error fetching prompt versions:", expect.any(Error));
    consoleError.mockRestore();
  });

  it("should select a version", async () => {
    const user = userEvent.setup();
    renderWithProviders(<VersionHistorySidePanel {...props} />);

    await user.click(await screen.findByRole("button", { name: /v1/i }));

    expect(props.onSelectVersion).toHaveBeenCalledWith(versions[1]);
  });

  it("should close from the dialog control", async () => {
    const user = userEvent.setup();
    renderWithProviders(<VersionHistorySidePanel {...props} />);

    await user.click(screen.getByRole("button", { name: "Close" }));

    expect(props.onClose).toHaveBeenCalledOnce();
  });

  it("should keep the surrounding editor interactive while history is open", async () => {
    const user = userEvent.setup();
    const onEditorAction = vi.fn();
    renderWithProviders(
      <>
        <button type="button" onClick={onEditorAction}>
          Editor action
        </button>
        <VersionHistorySidePanel {...props} />
      </>,
    );

    expect(await screen.findByRole("dialog", { name: "Version History" })).toHaveAttribute("aria-modal", "false");
    await user.click(screen.getByRole("button", { name: "Editor action" }));

    expect(onEditorAction).toHaveBeenCalledOnce();
  });

  it("should close on Escape", async () => {
    const user = userEvent.setup();
    renderWithProviders(<VersionHistorySidePanel {...props} />);
    await screen.findByText("v2");

    await user.keyboard("{Escape}");

    expect(props.onClose).toHaveBeenCalledOnce();
  });

  it("should keep history open when Escape dismisses a modal above it", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <>
        <VersionHistorySidePanel {...props} />
        <Dialog defaultOpen>
          <DialogContent>
            <DialogTitle>Prompt modal</DialogTitle>
          </DialogContent>
        </Dialog>
      </>,
    );
    await screen.findByRole("dialog", { name: "Prompt modal" });

    await user.keyboard("{Escape}");

    expect(props.onClose).not.toHaveBeenCalled();
  });
});
