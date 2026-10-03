import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import TaskClassifierSetup, { TASK_CLASSIFIER_ENDPOINT, type TaskClassifierSettings } from "./TaskClassifierSetup";
import { apiClient } from "@/components/networking";

vi.mock("@/components/networking", () => ({ apiClient: { get: vi.fn(), put: vi.fn() } }));

const jev = { id: "jev-id", name: "task-jev", provider: "typesafe" as const, model: "jev-latest" };
const laya = { id: "laya-id", name: "task-laya", provider: "laya" as const, model: "english" };

const settings = (overrides: Partial<TaskClassifierSettings> = {}): TaskClassifierSettings => ({
  enabled: false,
  model_id: null,
  models: [jev, laya],
  batch_size: 50,
  message_logging_enabled: true,
  ...overrides,
});

describe("TaskClassifierSetup", () => {
  beforeEach(() => {
    vi.mocked(apiClient.get).mockReset();
    vi.mocked(apiClient.put).mockReset();
  });

  it("says task classification has not been set up and saves the chosen System One model", async () => {
    vi.mocked(apiClient.get).mockResolvedValue(settings());
    vi.mocked(apiClient.put).mockResolvedValue(settings({ enabled: true, model_id: "laya-id" }));
    const onSaved = vi.fn();
    const user = userEvent.setup();
    render(<TaskClassifierSetup accessToken="token" canEdit onSaved={onSaved} />);

    expect(await screen.findByText("Task classification has not been set up")).toBeInTheDocument();
    await user.click(screen.getByRole("combobox", { name: "System One model" }));
    await user.click(await screen.findByRole("option", { name: /task-laya/ }));
    await user.click(screen.getByRole("button", { name: "Turn on task classification" }));

    expect(apiClient.put).toHaveBeenCalledWith(TASK_CLASSIFIER_ENDPOINT, {
      accessToken: "token",
      body: { enabled: true, model_id: "laya-id" },
    });
    expect(await screen.findByText(/Classifying untagged requests with/)).toHaveTextContent("task-laya");
    expect(onSaved).toHaveBeenCalledWith(expect.objectContaining({ enabled: true, model_id: "laya-id" }));
  });

  it("points to Models + Endpoints when no System One model is configured", async () => {
    vi.mocked(apiClient.get).mockResolvedValue(settings({ models: [] }));
    render(<TaskClassifierSetup accessToken="token" canEdit />);

    expect(await screen.findByRole("link", { name: "Models + Endpoints" })).toHaveAttribute(
      "href",
      expect.stringContaining("models-and-endpoints"),
    );
    expect(screen.queryByRole("button", { name: "Turn on task classification" })).not.toBeInTheDocument();
  });

  it("warns that nothing can be classified while prompt storage is off", async () => {
    vi.mocked(apiClient.get).mockResolvedValue(settings({ message_logging_enabled: false }));
    render(<TaskClassifierSetup accessToken="token" canEdit />);

    expect(await screen.findByText(/Prompt storage in spend logs is off/)).toBeInTheDocument();
  });

  it("does not offer setup controls to read-only admins", async () => {
    vi.mocked(apiClient.get).mockResolvedValue(settings());
    render(<TaskClassifierSetup accessToken="token" canEdit={false} />);

    expect(await screen.findByText("Ask a proxy admin to turn on task classification")).toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "System One model" })).not.toBeInTheDocument();
  });

  it("turns classification off from the enabled state", async () => {
    vi.mocked(apiClient.get).mockResolvedValue(settings({ enabled: true, model_id: "jev-id" }));
    vi.mocked(apiClient.put).mockResolvedValue(settings());
    const user = userEvent.setup();
    render(<TaskClassifierSetup accessToken="token" canEdit />);

    await user.click(await screen.findByRole("button", { name: "Turn off" }));

    expect(apiClient.put).toHaveBeenCalledWith(TASK_CLASSIFIER_ENDPOINT, {
      accessToken: "token",
      body: { enabled: false, model_id: null },
    });
    expect(await screen.findByText("Task classification has not been set up")).toBeInTheDocument();
  });

  it("explains when the saved model was removed", async () => {
    vi.mocked(apiClient.get).mockResolvedValue(settings({ enabled: true, model_id: "deleted" }));
    render(<TaskClassifierSetup accessToken="token" canEdit />);

    expect(await screen.findByText(/no longer configured/)).toBeInTheDocument();
  });

  it("shows the save error and keeps the form usable", async () => {
    vi.mocked(apiClient.get).mockResolvedValue(settings());
    vi.mocked(apiClient.put).mockRejectedValue(new Error("Select a configured System One model"));
    const user = userEvent.setup();
    render(<TaskClassifierSetup accessToken="token" canEdit />);

    await user.click(await screen.findByRole("button", { name: "Turn on task classification" }));

    expect(await screen.findByText("Select a configured System One model")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole("button", { name: "Turn on task classification" })).toBeEnabled());
  });
});
