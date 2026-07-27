import { renderWithProviders, screen, within } from "../../../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";
import GeneralSettings from "./general_settings";
import { deleteConfigFieldSetting, getGeneralSettingsCall, updateConfigFieldSetting } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  getGeneralSettingsCall: vi.fn(),
  updateConfigFieldSetting: vi.fn().mockResolvedValue({}),
  deleteConfigFieldSetting: vi.fn().mockResolvedValue({}),
}));

vi.mock("@/components/router_settings", () => ({ default: () => null }));
vi.mock("@/components/Settings/RouterSettings/Fallbacks/Fallbacks", () => ({ default: () => null }));
vi.mock("@/components/routing_groups", () => ({ default: () => null }));

// Mirrors the /config/list ordering: the two prompt-caching rows sit between the
// General-tab rows in the unfiltered response but are filtered out of the General
// tab's table, so any index-based lookup into the unfiltered array reads the wrong
// row for every field rendered after them.
const SETTINGS_FIXTURE = [
  {
    field_name: "budget_exceeded_throttle_percentage",
    field_type: "Float",
    field_value: null,
    field_description: "throttle fraction",
    stored_in_db: null,
    field_default_value: null,
  },
  {
    field_name: "enable_anthropic_prompt_caching",
    field_type: "Boolean",
    field_value: true,
    field_description: "prompt caching toggle",
    stored_in_db: true,
    field_tab: "prompt_caching",
    field_default_value: false,
  },
  {
    field_name: "anthropic_prompt_caching_ttl",
    field_type: "Select",
    field_value: "5m",
    field_description: "prompt caching ttl",
    stored_in_db: true,
    field_options: ["5m", "1h"],
    field_tab: "prompt_caching",
    field_default_value: null,
  },
  {
    field_name: "max_ui_session_budget",
    field_type: "Dollar",
    field_value: 7.5,
    field_description: "dashboard session budget",
    stored_in_db: true,
    field_default_value: 1.0,
  },
];

const settingsRow = async (fieldName: string) => {
  const cell = await screen.findByText(fieldName);
  const row = cell.closest("tr");
  expect(row).not.toBeNull();
  return row as HTMLElement;
};

describe("GeneralSettings General tab", () => {
  beforeEach(() => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([...SETTINGS_FIXTURE.map((s) => ({ ...s }))]);
    vi.mocked(updateConfigFieldSetting).mockClear();
    vi.mocked(deleteConfigFieldSetting).mockClear();
  });

  it("updates max_ui_session_budget with its own value, not the value at its filtered index", async () => {
    const user = userEvent.setup();
    renderWithProviders(<GeneralSettings accessToken="token" userRole="Admin" userID="user" />);

    await user.click(screen.getByText("General"));
    const row = await settingsRow("max_ui_session_budget");

    await user.click(within(row).getByRole("button", { name: /update/i }));

    expect(updateConfigFieldSetting).toHaveBeenCalledWith("token", "max_ui_session_budget", 7.5);
  });

  it("reset shows the field's default value instead of an empty input", async () => {
    const user = userEvent.setup();
    renderWithProviders(<GeneralSettings accessToken="token" userRole="Admin" userID="user" />);

    await user.click(screen.getByText("General"));
    const row = await settingsRow("max_ui_session_budget");
    expect(within(row).getByRole("spinbutton")).toHaveValue("7.50");

    const actionCell = row.querySelectorAll("td")[3];
    const resetIcon = actionCell.querySelector("svg");
    expect(resetIcon).not.toBeNull();
    await user.click(resetIcon as unknown as Element);

    expect(deleteConfigFieldSetting).toHaveBeenCalledWith("token", "max_ui_session_budget");
    expect(within(row).getByRole("spinbutton")).toHaveValue("1.00");
  });
});
