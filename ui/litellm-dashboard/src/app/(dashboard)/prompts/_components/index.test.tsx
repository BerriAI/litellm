import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getPromptsList } from "@/components/networking";

import PromptsPanel from "./index";

vi.mock("@/components/networking", () => ({
  getPromptsList: vi.fn(),
  deletePromptCall: vi.fn(),
}));

vi.mock("./PromptTable", () => ({
  __esModule: true,
  default: ({ isLoading }: { isLoading: boolean }) => (
    <div data-testid="prompt-table">{isLoading ? "table-loading" : "table-loaded"}</div>
  ),
}));

vi.mock("./prompt_info", () => ({ __esModule: true, default: () => null }));
vi.mock("./add_prompt_form", () => ({ __esModule: true, default: () => null }));
vi.mock("./prompt_editor_view", () => ({ __esModule: true, default: () => null }));

const mockGetPromptsList = vi.mocked(getPromptsList);

describe("PromptsPanel loading state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
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
      }),
    );
    render(<PromptsPanel accessToken="sk-test" userRole="Admin" />);
    expect(screen.getByText("table-loading")).toBeInTheDocument();

    resolveFetch({ prompts: [] });
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockGetPromptsList).toHaveBeenCalledWith("sk-test", undefined);
  });
});
