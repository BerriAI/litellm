import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import VersionHistorySidePanel from "./VersionHistorySidePanel";
import { getPromptVersions } from "@/components/networking";

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
    prompt_info: { prompt_type: "db" },
  },
];

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
    vi.mocked(getPromptVersions).mockResolvedValue({ prompts: versions } as any);
  });

  it("loads and renders prompt versions", async () => {
    render(<VersionHistorySidePanel {...props} />);
    expect(await screen.findByText("v2")).toBeInTheDocument();
    expect(screen.getByText("v1")).toBeInTheDocument();
    expect(screen.getByText("Latest")).toBeInTheDocument();
    expect(screen.getByText("Active")).toBeInTheDocument();
  });

  it("selects a version", async () => {
    render(<VersionHistorySidePanel {...props} />);
    fireEvent.click(await screen.findByText("v1"));
    expect(props.onSelectVersion).toHaveBeenCalledWith(versions[1]);
  });

  it("shows an empty result", async () => {
    vi.mocked(getPromptVersions).mockResolvedValue({ prompts: [] } as any);
    render(<VersionHistorySidePanel {...props} />);
    expect(await screen.findByText("No version history available.")).toBeInTheDocument();
  });

  it("closes from the dialog control", async () => {
    render(<VersionHistorySidePanel {...props} />);
    await screen.findByText("v2");
    const buttons = screen.getAllByRole("button");
    fireEvent.click(buttons[0]);
    expect(props.onClose).toHaveBeenCalled();
  });

  it("keeps the surrounding editor interactive while history is open", async () => {
    const onEditorAction = vi.fn();
    render(
      <>
        <button type="button" onClick={onEditorAction}>
          Editor action
        </button>
        <VersionHistorySidePanel {...props} />
      </>,
    );
    await screen.findByText("v2");
    expect(screen.getByRole("dialog", { name: "Version History" })).toHaveAttribute("aria-modal", "false");
    fireEvent.click(screen.getByRole("button", { name: "Editor action" }));
    expect(onEditorAction).toHaveBeenCalledOnce();
  });

  it("closes on Escape", async () => {
    render(<VersionHistorySidePanel {...props} />);
    await screen.findByText("v2");
    fireEvent.keyDown(document, { key: "Escape" });
    expect(props.onClose).toHaveBeenCalledOnce();
  });

  it("does not load while closed", async () => {
    await act(async () => render(<VersionHistorySidePanel {...props} isOpen={false} />));
    await waitFor(() => expect(getPromptVersions).not.toHaveBeenCalled());
  });
});
