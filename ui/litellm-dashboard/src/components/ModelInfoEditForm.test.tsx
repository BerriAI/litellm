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
    credential_alias: "Main OpenAI",
    credential_values: {},
    credential_info: { custom_llm_provider: "openai" },
  },
];

const renderEditing = (onSubmit = vi.fn().mockResolvedValue(undefined)) =>
  render(
    <ModelInfoEditForm
      localModelData={{
        model_name: "gpt-4o",
        litellm_params: { model: "gpt-4o" },
        model_info: {},
      }}
      modelData={{ model_info: {} }}
      teamAlias={null}
      accessToken="test-token"
      isEditing={true}
      isSaving={false}
      isWildcardModel={false}
      ptuCostAttributionEnabled={false}
      showCacheControl={false}
      setShowCacheControl={vi.fn()}
      onCancel={vi.fn()}
      onSubmit={onSubmit}
      modelAccessGroups={[]}
      guardrailsList={[]}
      tagsList={{}}
      credentialsList={credentialsList}
      healthCheckModelOptions={[]}
      teams={[]}
    />,
  );

describe("ModelInfoEditForm existing-credentials picker", () => {
  it("shows the alias beside the name, searches by alias, and submits the name", async () => {
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderEditing(onSubmit);

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
});
