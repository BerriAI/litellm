import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi, type Mock } from "vitest";

import { FairnessSettingsForm } from "./FairnessSettingsForm";
import type { FairnessSettings, FairnessSettingsResponse } from "./schema";

const SETTINGS: FairnessSettings = {
  enabled: false,
  workload_classes: [
    { name: "production", reserved_share: 0.5, max_queue_wait_seconds: 30, description: "customer traffic" },
  ],
  default_reserved_share: 0.25,
  default_max_queue_wait_seconds: 0,
  saturation_threshold: 0.5,
  saturation_check_cache_ttl: 60,
  max_queue_depth_per_class: 100,
  queue_poll_interval_seconds: 0.1,
};

const RESPONSE: FairnessSettingsResponse = { settings: SETTINGS, persisted: true };

const renderForm = (overrides?: {
  response?: FairnessSettingsResponse;
  fetchSettings?: Mock;
  updateSettings?: Mock;
  readOnly?: boolean;
}) => {
  const fetchSettings = overrides?.fetchSettings ?? vi.fn().mockResolvedValue(overrides?.response ?? RESPONSE);
  const updateSettings = overrides?.updateSettings ?? vi.fn().mockResolvedValue(undefined);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

  render(
    <QueryClientProvider client={queryClient}>
      <FairnessSettingsForm
        readOnly={overrides?.readOnly ?? false}
        fetchSettings={fetchSettings}
        updateSettings={updateSettings}
      />
    </QueryClientProvider>,
  );

  return { fetchSettings, updateSettings };
};

const saveButton = async () => await screen.findByRole("button", { name: "Save settings" });

describe("FairnessSettingsForm", () => {
  it("shows the loaded settings as percentages and keeps Save disabled until edited", async () => {
    renderForm();

    expect(await screen.findByLabelText("Saturation threshold (%)")).toHaveValue(50);
    expect(screen.getByLabelText("Default pool reserved share (%)")).toHaveValue(25);
    expect(screen.getByLabelText("Reserved share (%)")).toHaveValue(50);
    expect(screen.getByLabelText("Name")).toHaveValue("production");
    expect(screen.getByRole("switch", { name: "Enable fairness under load" })).not.toBeChecked();
    expect(screen.getByTestId("share-summary")).toHaveTextContent("Reserved in total: 75%");
    expect(await saveButton()).toBeDisabled();
  });

  it("sends the whole settings object with shares converted back to fractions", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await user.click(await screen.findByRole("switch", { name: "Enable fairness under load" }));
    fireEvent.change(screen.getByLabelText("Saturation threshold (%)"), { target: { value: "80" } });
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({ ...SETTINGS, enabled: true, saturation_threshold: 0.8 });
  });

  it("adds a workload class and sends it along", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await user.click(await screen.findByRole("button", { name: "Add workload class" }));
    const second = within(screen.getByTestId("workload-class-1"));
    fireEvent.change(second.getByLabelText("Name"), { target: { value: "batch" } });
    fireEvent.change(second.getByLabelText("Reserved share (%)"), { target: { value: "10" } });
    fireEvent.change(second.getByLabelText("Max queue wait (s)"), { target: { value: "120" } });
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({
      ...SETTINGS,
      workload_classes: [
        ...SETTINGS.workload_classes,
        { name: "batch", reserved_share: 0.1, max_queue_wait_seconds: 120, description: null },
      ],
    });
  });

  it("blocks saving when the reserved shares exceed 100% and explains why", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    fireEvent.change(await screen.findByLabelText("Default pool reserved share (%)"), { target: { value: "60" } });
    await user.click(await saveButton());

    expect(
      await screen.findByText("Reserved shares add up to 110%. Keep the total at or below 100%"),
    ).toBeInTheDocument();
    expect(updateSettings).not.toHaveBeenCalled();
  });

  it("rejects the reserved default class name", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    fireEvent.change(await screen.findByLabelText("Name"), { target: { value: "default" } });
    await user.click(await saveButton());

    expect(await screen.findByText(/reserved for/)).toBeInTheDocument();
    expect(updateSettings).not.toHaveBeenCalled();
  });

  it("hides Save and disables every control for viewers", async () => {
    renderForm({ readOnly: true });

    expect(await screen.findByLabelText("Saturation threshold (%)")).toBeDisabled();
    expect(screen.getByRole("switch", { name: "Enable fairness under load" })).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("button", { name: "Add workload class" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Save settings" })).not.toBeInTheDocument();
  });

  it("tells admins when the values still come from config.yaml", async () => {
    renderForm({ response: { settings: SETTINGS, persisted: false } });

    expect(await screen.findByText(/values from config.yaml/)).toBeInTheDocument();
  });

  it("shows an error when the settings cannot be loaded", async () => {
    renderForm({ fetchSettings: vi.fn().mockRejectedValue(new Error("nope")) });

    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load the fairness settings.");
  });
});
