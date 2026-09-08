import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import PluginSettings from "./PluginSettings";

const { getConfigFieldSettingMock, updateConfigFieldSettingMock } = vi.hoisted(() => ({
  getConfigFieldSettingMock: vi.fn(),
  updateConfigFieldSettingMock: vi.fn(),
}));

vi.mock("@/components/networking", () => ({
  getConfigFieldSetting: getConfigFieldSettingMock,
  updateConfigFieldSetting: updateConfigFieldSettingMock,
}));

const REDACTED_PLUGIN = {
  name: "alpha",
  display_name: "Alpha",
  url: "https://alpha.example.com",
  plugin_key: "***",
};

const savedPayload = () => updateConfigFieldSettingMock.mock.calls[0];

describe("PluginSettings config payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    updateConfigFieldSettingMock.mockResolvedValue({});
  });

  it("sends a new plugin with no plugin_key when the key field is left blank", async () => {
    const user = userEvent.setup();
    getConfigFieldSettingMock.mockResolvedValue({ field_value: [] });
    render(<PluginSettings />);
    expect(await screen.findByText("No data", { ignore: "title" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /add plugin/i }));
    fireEvent.change(await screen.findByLabelText(/Name \(identifier\)/), { target: { value: "beta" } });
    fireEvent.change(screen.getByLabelText(/Display Name/), { target: { value: "Beta" } });
    fireEvent.change(screen.getByLabelText(/^URL/), { target: { value: "https://beta.example.com" } });
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(updateConfigFieldSettingMock).toHaveBeenCalledTimes(1));
    expect(savedPayload()).toStrictEqual([
      "123",
      "plugins",
      [
        {
          name: "beta",
          display_name: "Beta",
          url: "https://beta.example.com",
          plugin_key: undefined,
        },
      ],
    ]);
  });

  it("seeds the key field blank on edit and sends a blank key when it is left untouched", async () => {
    const user = userEvent.setup();
    getConfigFieldSettingMock.mockResolvedValue({ field_value: [REDACTED_PLUGIN] });
    render(<PluginSettings />);
    expect(await screen.findByText("Alpha")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Edit alpha" }));
    expect(await screen.findByLabelText(/Plugin Key/)).toHaveValue("");

    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(updateConfigFieldSettingMock).toHaveBeenCalledTimes(1));
    expect(savedPayload()).toStrictEqual([
      "123",
      "plugins",
      [
        {
          name: "alpha",
          display_name: "Alpha",
          url: "https://alpha.example.com",
          plugin_key: "",
        },
      ],
    ]);
  });

  it("sends the typed key on edit when the key field is filled in", async () => {
    const user = userEvent.setup();
    getConfigFieldSettingMock.mockResolvedValue({ field_value: [REDACTED_PLUGIN] });
    render(<PluginSettings />);
    expect(await screen.findByText("Alpha")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Edit alpha" }));
    fireEvent.change(await screen.findByLabelText(/Plugin Key/), { target: { value: "sk-brand-new" } });
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(updateConfigFieldSettingMock).toHaveBeenCalledTimes(1));
    expect(savedPayload()).toStrictEqual([
      "123",
      "plugins",
      [
        {
          name: "alpha",
          display_name: "Alpha",
          url: "https://alpha.example.com",
          plugin_key: "sk-brand-new",
        },
      ],
    ]);
  });
});

describe("PluginSettings plugin key reveal (post-migration shadcn affordance)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getConfigFieldSettingMock.mockResolvedValue({ field_value: [REDACTED_PLUGIN] });
  });

  it("flips the key field between hidden and revealed and relabels the toggle", async () => {
    const user = userEvent.setup();
    render(<PluginSettings />);
    expect(await screen.findByText("Alpha")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Edit alpha" }));
    const keyInput = await screen.findByLabelText(/Plugin Key/);
    expect(keyInput).toHaveAttribute("type", "password");

    await user.click(screen.getByRole("button", { name: "Show plugin key" }));
    expect(keyInput).toHaveAttribute("type", "text");
    expect(screen.queryByRole("button", { name: "Show plugin key" })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Hide plugin key" }));
    expect(keyInput).toHaveAttribute("type", "password");
    expect(screen.queryByRole("button", { name: "Hide plugin key" })).not.toBeInTheDocument();
  });
});
