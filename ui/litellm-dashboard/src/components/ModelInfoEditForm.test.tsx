import { render, screen } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import type { CredentialItem } from "@/components/networking";

import ModelInfoEditForm from "./ModelInfoEditForm";

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    vectorStoreListCall: vi.fn().mockResolvedValue({ data: [] }),
  };
});

const credentialsList: CredentialItem[] = [
  {
    credential_name: "openai-main",
    display_name: "Main OpenAI",
    credential_values: {},
    credential_info: { custom_llm_provider: "openai" },
  },
];

const renderForm = ({
  onSubmit = vi.fn().mockResolvedValue(undefined),
  isEditing = true,
  litellmParams = { model: "gpt-4o" } as Record<string, unknown>,
} = {}) =>
  render(
    <ModelInfoEditForm
      localModelData={{
        model_name: "gpt-4o",
        litellm_params: litellmParams,
        model_info: {},
      }}
      modelData={{ model_info: {} }}
      teamAlias={null}
      accessToken="test-token"
      isEditing={isEditing}
      isSaving={false}
      isWildcardModel={false}
      ptuCostAttributionEnabled={false}
      showCacheControl={false}
      setShowCacheControl={vi.fn()}
      onCancel={vi.fn()}
      onSubmit={onSubmit}
      modelAccessGroups={[]}
      guardrailsList={[]}
      tagsList={[]}
      credentialsList={credentialsList}
      healthCheckModelOptions={[]}
      teams={[]}
    />,
  );

describe("ModelInfoEditForm existing-credentials picker", () => {
  it("shows the display name beside the credential name, searches by it, and submits the name", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderForm({ onSubmit });

    const picker = await screen.findByPlaceholderText("Select or search for existing credentials");
    await user.click(picker);

    const option = await screen.findByRole("option", { name: /openai-main/ });
    expect(option).toHaveTextContent("openai-main");
    expect(option).toHaveTextContent("Main OpenAI");

    await user.type(picker, "main openai");
    await user.click(await screen.findByRole("option", { name: /openai-main/ }));
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await vi.waitFor(() => expect(onSubmit).toHaveBeenCalled());
    expect(onSubmit.mock.calls[0][0].litellm_credential_name).toBe("openai-main");
  });

  it("keeps the attached credential when the search text is emptied and only clears through None", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderForm({ onSubmit, litellmParams: { model: "gpt-4o", litellm_credential_name: "openai-main" } });

    const picker = await screen.findByPlaceholderText("Select or search for existing credentials");
    expect(picker).toHaveValue("Main OpenAI");
    await user.clear(picker);
    await user.tab();
    await user.click(screen.getByRole("button", { name: /save changes/i }));
    await vi.waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0].litellm_credential_name).toBe("openai-main");

    await user.click(picker);
    await user.click(await screen.findByRole("option", { name: "None" }));
    await user.click(screen.getByRole("button", { name: /save changes/i }));
    await vi.waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(2));
    expect(onSubmit.mock.calls[1][0].litellm_credential_name).toBeNull();
  });

  it("shows the attached credential's display name when not editing", () => {
    renderForm({ isEditing: false, litellmParams: { model: "gpt-4o", litellm_credential_name: "openai-main" } });

    expect(screen.getByText("Main OpenAI")).toBeInTheDocument();
    expect(screen.queryByText("openai-main")).not.toBeInTheDocument();
  });

  it("falls back to the credential name, then Manual, when not editing", () => {
    const { unmount } = renderForm({
      isEditing: false,
      litellmParams: { model: "gpt-4o", litellm_credential_name: "not-listed" },
    });
    expect(screen.getByText("not-listed")).toBeInTheDocument();
    unmount();

    renderForm({ isEditing: false });
    expect(screen.getByText("Manual")).toBeInTheDocument();
  });
});
