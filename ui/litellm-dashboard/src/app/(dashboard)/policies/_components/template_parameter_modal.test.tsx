import React from "react";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "@/../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import TemplateParameterModal from "./template_parameter_modal";
import * as networking from "@/components/networking";

vi.mock("@/components/networking", () => ({
  modelHubCall: vi.fn(),
  enrichPolicyTemplateStream: vi.fn(),
}));

const template = {
  id: "competitor-blocking",
  title: "Competitor Blocking",
  parameters: [
    {
      name: "brand_name",
      label: "Brand Name",
      type: "string",
      required: true,
      placeholder: "e.g. Acme Airlines",
    },
  ],
  llm_enrichment: { parameter: "brand_name" },
};

const defaultProps = {
  visible: true,
  template,
  onConfirm: vi.fn(),
  onCancel: vi.fn(),
  accessToken: "test-token",
};

type UserEvent = ReturnType<typeof userEvent.setup>;

const startEnrichment = async (user: UserEvent) => {
  await user.type(screen.getByPlaceholderText("e.g. Acme Airlines"), "Acme Airlines");

  const modelSection = screen.getByText("Select Model").closest("div") as HTMLElement;
  const combobox = within(modelSection).getByRole("combobox");
  await user.click(combobox);
  await user.click(await screen.findByTitle("gpt-4o"));

  await user.click(screen.getByRole("button", { name: /generate competitor names/i }));

  await waitFor(() => {
    expect(networking.enrichPolicyTemplateStream).toHaveBeenCalledTimes(1);
  });

  const options = vi.mocked(networking.enrichPolicyTemplateStream).mock.calls[0][7];
  expect(options?.signal).toBeInstanceOf(AbortSignal);
  return options!.signal!;
};

describe("TemplateParameterModal enrichment stream cancellation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.modelHubCall).mockResolvedValue({
      data: [{ model_group: "gpt-4o" }],
    } as any);
    vi.mocked(networking.enrichPolicyTemplateStream).mockReturnValue(new Promise(() => {}));
  });

  it("aborts an in-flight enrichment stream when the modal unmounts", async () => {
    const user = userEvent.setup();
    const { unmount } = renderWithProviders(<TemplateParameterModal {...defaultProps} />);

    const signal = await startEnrichment(user);
    expect(signal.aborted).toBe(false);

    unmount();

    expect(signal.aborted).toBe(true);
  });

  it("aborts an in-flight enrichment stream when the modal is closed", async () => {
    const user = userEvent.setup();
    const { rerender } = renderWithProviders(<TemplateParameterModal {...defaultProps} />);

    const signal = await startEnrichment(user);
    expect(signal.aborted).toBe(false);

    rerender(<TemplateParameterModal {...defaultProps} visible={false} />);

    await waitFor(() => {
      expect(signal.aborted).toBe(true);
    });
  });
});
