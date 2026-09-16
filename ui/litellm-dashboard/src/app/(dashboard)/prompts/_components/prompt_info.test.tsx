import React from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import PromptInfoView from "./prompt_info";
import * as networking from "@/components/networking";

vi.mock("@/components/networking", () => ({
  getPromptInfo: vi.fn(),
  getPromptVersions: vi.fn(),
  deletePromptCall: vi.fn(),
}));

vi.mock("./prompt_editor_view/PromptCodeSnippets", () => ({
  default: ({ environment }: { environment?: string }) => (
    <div data-testid="prompt-code-snippets" data-environment={environment} />
  ),
}));

const promptWithoutTemplate = {
  prompt_spec: {
    prompt_id: "support-reply",
    version: 1,
    litellm_params: { prompt_id: "support-reply" },
    prompt_info: { prompt_type: "dotprompt" },
    created_by: "admin",
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
  },
  raw_prompt_template: null,
  environments: [],
};

const promptWithTemplate = {
  ...promptWithoutTemplate,
  prompt_spec: { ...promptWithoutTemplate.prompt_spec, version: 3, environment: "development" },
  raw_prompt_template: { litellm_prompt_id: "support-reply", content: "Hello {{name}}" },
  environments: ["development", "production"],
};

const versionRow = (version: number) => ({
  ...promptWithoutTemplate.prompt_spec,
  prompt_id: `support-reply.v${version}`,
  version,
  environment: "development",
});

const renderInfo = (searchParams: Record<string, string>, onUrlUpdate?: OnUrlUpdateFunction) =>
  renderWithProviders(
    <PromptInfoView
      promptId="support-reply"
      initialEnvironment="development"
      onClose={vi.fn()}
      accessToken="sk-test"
      isAdmin={true}
    />,
    { searchParams, onUrlUpdate },
  );

const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) => onUrlUpdate.mock.calls.at(-1)?.[0];

describe("PromptInfoView environment scoping", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.mocked(networking.getPromptInfo).mockReset().mockResolvedValue(promptWithoutTemplate);
    vi.mocked(networking.getPromptVersions).mockReset().mockResolvedValue({ prompts: [] });
  });

  it("fetches the initial environment it was opened with", async () => {
    renderWithProviders(
      <PromptInfoView
        promptId="support-reply"
        initialEnvironment="staging"
        onClose={vi.fn()}
        accessToken="sk-test"
        isAdmin={true}
      />,
    );

    await screen.findByRole("tab", { name: "Raw JSON" });
    expect(networking.getPromptInfo).toHaveBeenCalledWith("sk-test", "support-reply", "staging");
  });

  it("fetches the serve default when opened without an environment", async () => {
    renderWithProviders(
      <PromptInfoView promptId="support-reply" onClose={vi.fn()} accessToken="sk-test" isAdmin={true} />,
    );

    await screen.findByRole("tab", { name: "Raw JSON" });
    expect(networking.getPromptInfo).toHaveBeenCalledWith("sk-test", "support-reply", undefined);
  });
});

describe("PromptInfoView code snippets", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.mocked(networking.getPromptVersions).mockReset().mockResolvedValue({ prompts: [] });
  });

  it.each([
    ["a prompt with several environments", "staging", ["development", "staging"]],
    ["a config prompt with no environment list", "development", []],
  ])("hands the viewed environment of %s to the code snippets", async (_label, environment, environments) => {
    vi.mocked(networking.getPromptInfo)
      .mockReset()
      .mockResolvedValue({
        ...promptWithoutTemplate,
        prompt_spec: { ...promptWithoutTemplate.prompt_spec, environment },
        environments,
      });

    renderWithProviders(
      <PromptInfoView
        promptId="support-reply"
        initialEnvironment={environment}
        onClose={vi.fn()}
        accessToken="sk-test"
        isAdmin={true}
      />,
    );

    await screen.findByRole("tab", { name: "Raw JSON" });
    expect(screen.getByTestId("prompt-code-snippets")).toHaveAttribute("data-environment", environment);
  });
});

describe("PromptInfoView tabs", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.mocked(networking.getPromptInfo).mockReset().mockResolvedValue(promptWithoutTemplate);
    vi.mocked(networking.getPromptVersions).mockReset().mockResolvedValue({ prompts: [] });
  });

  it("shows the raw API response for a prompt that has no template", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(
      <PromptInfoView promptId="support-reply" onClose={vi.fn()} accessToken="sk-test" isAdmin={true} />,
      { onUrlUpdate },
    );

    expect(await screen.findByRole("tab", { name: "Raw JSON" })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Prompt Template" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Raw JSON" }));

    expect(screen.getByText("Raw API Response")).toBeVisible();
    expect(screen.getByText(/"prompt_id": "support-reply"/)).toBeVisible();
    await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("tab")).toBe("raw"));
  });

  it("opens on the tab in the URL", async () => {
    renderInfo({ tab: "raw" });

    expect(await screen.findByRole("tab", { name: "Raw JSON" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Raw API Response")).toBeVisible();
  });

  it("opens the template tab from the URL when the prompt has a template", async () => {
    vi.mocked(networking.getPromptInfo).mockResolvedValue(promptWithTemplate);
    renderInfo({ tab: "template" });

    expect(await screen.findByRole("tab", { name: "Prompt Template" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Hello {{name}}")).toBeVisible();
  });

  it("falls back to the overview and clears tab=template when the prompt has no template", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    render(
      <NuqsTestingAdapter
        searchParams={{ tab: "template" }}
        onUrlUpdate={onUrlUpdate}
        hasMemory
        resetUrlUpdateQueueOnMount={false}
      >
        <QueryClientProvider client={testQueryClient}>
          <PromptInfoView promptId="support-reply" onClose={vi.fn()} accessToken="sk-test" isAdmin={true} />
        </QueryClientProvider>
      </NuqsTestingAdapter>,
    );

    expect(await screen.findByRole("tab", { name: "Overview" })).toHaveAttribute("aria-selected", "true");
    await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.has("tab")).toBe(false));
    expect(onUrlUpdate).toHaveBeenCalled();
  });
});

describe("PromptInfoView versions and environments in the URL", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.mocked(networking.getPromptInfo).mockReset().mockResolvedValue(promptWithTemplate);
    vi.mocked(networking.getPromptVersions)
      .mockReset()
      .mockResolvedValue({ prompts: [versionRow(1), versionRow(2), versionRow(3)] });
  });

  it("loads the version named in the URL", async () => {
    vi.mocked(networking.getPromptInfo).mockResolvedValue({
      ...promptWithTemplate,
      prompt_spec: { ...promptWithTemplate.prompt_spec, version: 2 },
    });
    renderInfo({ version: "2", prompt_env: "development" });

    expect(await screen.findByText("Viewing v2 — not the latest version (v3)")).toBeInTheDocument();
    expect(networking.getPromptInfo).toHaveBeenCalledWith("sk-test", "support-reply.v2", "development");
    expect(networking.getPromptInfo).not.toHaveBeenCalledWith("sk-test", "support-reply", "development");
  });

  it("loads the environment in the URL over the one it was opened with", async () => {
    renderInfo({ prompt_env: "production" });

    await screen.findByRole("tab", { name: "Raw JSON" });
    expect(networking.getPromptInfo).toHaveBeenCalledWith("sk-test", "support-reply", "production");
    expect(networking.getPromptInfo).not.toHaveBeenCalledWith("sk-test", "support-reply", "development");
    await waitFor(() =>
      expect(networking.getPromptVersions).toHaveBeenCalledWith("sk-test", "support-reply", "production"),
    );
  });

  it("writes the clicked version to the URL and loads it", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderInfo({}, onUrlUpdate);

    await user.click(await screen.findByText("v2"));

    await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("version")).toBe("2"));
    expect(lastUrl(onUrlUpdate)?.searchParams.get("prompt_env")).toBe("development");
    await waitFor(() =>
      expect(networking.getPromptInfo).toHaveBeenCalledWith("sk-test", "support-reply.v2", "development"),
    );
  });

  it("writes the clicked environment to the URL and drops the version", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderInfo({ version: "2", prompt_env: "development" }, onUrlUpdate);

    await user.click(await screen.findByRole("button", { name: "production" }));

    await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("prompt_env")).toBe("production"));
    expect(lastUrl(onUrlUpdate)?.searchParams.has("version")).toBe(false);
    await waitFor(() =>
      expect(networking.getPromptInfo).toHaveBeenCalledWith("sk-test", "support-reply", "production"),
    );
  });

  it("drops the version from the URL when going back to the latest", async () => {
    vi.mocked(networking.getPromptInfo).mockResolvedValue({
      ...promptWithTemplate,
      prompt_spec: { ...promptWithTemplate.prompt_spec, version: 2 },
    });
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderInfo({ version: "2", prompt_env: "development" }, onUrlUpdate);

    await user.click(await screen.findByRole("button", { name: "Go to latest" }));

    await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.has("version")).toBe(false));
    expect(lastUrl(onUrlUpdate)?.searchParams.get("prompt_env")).toBe("development");
  });
});
