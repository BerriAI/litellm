import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import LanguageSelector from "@/components/LanguageSelector";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { translate, type Language } from "@/i18n/language";
import { Toaster } from "@/components/ui/sonner";
import { ApiError } from "@/lib/http/client";

vi.unmock("@/lib/toast");

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useInfiniteTeams: () => ({
    data: {
      pages: [
        {
          teams: [
            { team_id: "team-alpha", team_alias: "Alpha" },
            { team_id: "team-beta", team_alias: "Beta" },
          ],
        },
      ],
    },
    fetchNextPage: vi.fn(),
    hasNextPage: false,
    isFetchingNextPage: false,
    isLoading: false,
  }),
}));

vi.mock("@/components/ModelSelect/ModelSelect", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/ModelSelect/ModelSelect")>();
  return {
    MODEL_SENTINEL_OPTIONS: actual.MODEL_SENTINEL_OPTIONS,
    ModelSelect: ({ onChange }: { onChange: (values: string[]) => void }) => (
      <button type="button" onClick={() => onChange(["all-proxy-models"])}>
        set-models
      </button>
    ),
  };
});

import { toast } from "@/lib/toast";

import { DefaultUserSettingsForm } from "./DefaultUserSettingsForm";
import type { InternalUserSettings } from "./mapper";

const POSSIBLE_UI_ROLES = {
  internal_user: { ui_label: "Internal User", description: "create and view own keys" },
  internal_user_viewer: { ui_label: "Internal Viewer", description: "view own keys" },
  proxy_admin: { ui_label: "Admin", description: "all permissions" },
};

const SETTINGS: InternalUserSettings = {
  values: {
    user_role: "internal_user",
    max_budget: 100,
    budget_duration: "30d",
    models: ["gpt-5.2"],
    teams: [{ team_id: "team-alpha", max_budget_in_team: 25, user_role: "user" }],
  },
  field_schema: {},
};

const SAVED_BODY = {
  user_role: "internal_user",
  max_budget: 100,
  budget_duration: "30d",
  models: ["gpt-5.2"],
  teams: [{ team_id: "team-alpha", max_budget_in_team: 25, user_role: "user" }],
};

const renderForm = (overrides?: {
  fetchSettings?: Mock;
  updateSettings?: Mock;
  language?: Language;
  roles?: typeof POSSIBLE_UI_ROLES;
}) => {
  const fetchSettings = overrides?.fetchSettings ?? vi.fn().mockResolvedValue(SETTINGS);
  const updateSettings = overrides?.updateSettings ?? vi.fn().mockResolvedValue(undefined);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });

  render(
    <QueryClientProvider client={queryClient}>
      <LanguageProvider
        defaultLanguage={overrides?.language ?? "en"}
        userId="default-users"
        storageScope="default-users"
      >
        <LanguageSelector />
        <DefaultUserSettingsForm
          possibleUIRoles={overrides?.roles ?? POSSIBLE_UI_ROLES}
          fetchSettings={fetchSettings}
          updateSettings={updateSettings}
        />
        <Toaster expand />
      </LanguageProvider>
    </QueryClientProvider>,
  );

  return { fetchSettings, updateSettings };
};

const saveButton = async () => await screen.findByRole("button", { name: "Save Changes" });

const enterEditMode = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(await screen.findByRole("button", { name: "Edit Settings" }));
};

describe("DefaultUserSettingsForm", () => {
  beforeEach(() => {
    localStorage.clear();
    toast.dismiss();
    vi.clearAllMocks();
  });

  afterEach(() => act(() => toast.dismiss()));

  it.each<Language>(["en", "zh-CN"])("默认用户的嵌套校验随 %s 切换并保留团队角色", async (initialLanguage) => {
    const user = userEvent.setup();
    const nextLanguage = initialLanguage === "en" ? "zh-CN" : "en";
    const { fetchSettings, updateSettings } = renderForm({ language: initialLanguage });
    await user.click(await screen.findByRole("button", { name: translate(initialLanguage, "Edit Settings") }));
    fireEvent.change(screen.getByLabelText(translate(initialLanguage, "Max Budget (USD)")), {
      target: { value: "-1" },
    });
    fireEvent.change(screen.getByLabelText(translate(initialLanguage, "Max Budget in Team (USD)")), {
      target: { value: "-2" },
    });
    await user.click(screen.getByRole("button", { name: translate(initialLanguage, "Add Team") }));
    await user.click(screen.getAllByLabelText(translate(initialLanguage, "Team"), { exact: true })[1]);
    await user.click(await screen.findByText("Alpha"));
    await user.click(screen.getByRole("button", { name: translate(initialLanguage, "Save Changes") }));
    expect(await screen.findByText(translate(initialLanguage, "This team is already listed"))).toBeVisible();

    // 保留已选团队和金额草稿，不通过重新加载设置更新文案。
    fireEvent.change(screen.getByLabelText(translate(initialLanguage, "Interface language")), {
      target: { value: nextLanguage },
    });
    expect(screen.getByText(translate(nextLanguage, "This team is already listed"))).toBeVisible();
    expect(screen.getAllByText(translate(nextLanguage, "Must be a non-negative number"))).toHaveLength(2);
    expect(screen.getByLabelText(translate(nextLanguage, "Max Budget (USD)"))).toHaveValue(-1);
    expect(screen.getAllByLabelText(translate(nextLanguage, "Team"), { exact: true })[1]).toHaveValue("Alpha");
    expect(fetchSettings).toHaveBeenCalledTimes(1);
    expect(updateSettings).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText(translate(nextLanguage, "Max Budget (USD)")), { target: { value: "250" } });
    fireEvent.change(screen.getAllByLabelText(translate(nextLanguage, "Max Budget in Team (USD)"))[0], {
      target: { value: "30" },
    });
    await user.click(screen.getAllByLabelText(translate(nextLanguage, "Team"), { exact: true })[1]);
    await user.click(await screen.findByText("Beta"));
    await user.click(screen.getByRole("button", { name: translate(nextLanguage, "Save Changes") }));
    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    const expected = {
      ...SAVED_BODY,
      max_budget: 250,
      teams: [
        { team_id: "team-alpha", max_budget_in_team: 30, user_role: "user" },
        { team_id: "team-beta", max_budget_in_team: null, user_role: "user" },
      ],
    };
    expect(updateSettings).toHaveBeenCalledWith(expected);
  });

  it("默认用户缺失团队的已有提示切换语言且不清空新增行", async () => {
    const user = userEvent.setup();
    const { fetchSettings, updateSettings } = renderForm({ language: "zh-CN" });
    await user.click(await screen.findByRole("button", { name: translate("zh-CN", "Edit Settings") }));
    await user.click(screen.getByRole("button", { name: translate("zh-CN", "Add Team") }));
    await user.click(screen.getByRole("button", { name: translate("zh-CN", "Save Changes") }));
    expect(await screen.findByText(translate("zh-CN", "Select a team"))).toBeVisible();
    fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
    expect(screen.getByText("Select a team")).toBeVisible();
    expect(screen.getAllByLabelText("Team", { exact: true })).toHaveLength(2);
    expect(fetchSettings).toHaveBeenCalledTimes(1);
    expect(updateSettings).not.toHaveBeenCalled();
  });

  it.each([
    ["Internal User", "内部用户"],
    ["Home", "Home"],
  ])("默认角色标签 %s 在摘要和选项保持同一语言边界", async (label, expected) => {
    const user = userEvent.setup();
    const roles = { ...POSSIBLE_UI_ROLES, internal_user: { ...POSSIBLE_UI_ROLES.internal_user, ui_label: label } };
    renderForm({ language: "zh-CN", roles });
    expect(await screen.findByText(expected, { exact: true })).toBeVisible();
    await user.click(screen.getByRole("button", { name: translate("zh-CN", "Edit Settings") }));
    const selector = screen.getByLabelText(translate("zh-CN", "Default Role"));
    expect(selector).toHaveTextContent(expected);
    await user.click(selector);
    expect(await screen.findByRole("option", { name: new RegExp(`^${expected}`) })).toBeVisible();
    expect(screen.queryByText("首页")).not.toBeInTheDocument();
  });

  it.each(["success", "failure", "empty", "server", "string", "envelope"])(
    "默认用户保存的 %s 通知切换语言且不改保存载荷",
    async (result) => {
      const user = userEvent.setup();
      const request = Promise.withResolvers<void>();
      const updateSettings = vi.fn().mockReturnValue(request.promise);
      renderForm({ updateSettings });
      await enterEditMode(user);
      fireEvent.change(screen.getByLabelText("Max Budget (USD)"), { target: { value: "250" } });
      await user.click(await saveButton());
      await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
      fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
      await act(async () => {
        if (result === "success") request.resolve();
        else if (result === "server") request.reject(new ApiError("Home", 403, { error: { message: "Home" } }));
        else if (result === "string") request.reject("Home");
        else if (result === "envelope") request.reject({ error: { message: "Home", code: 403 } });
        else request.reject(result === "empty" ? new Error("") : null);
      });
      const message =
        result === "success" ? "Default user settings updated successfully" : "Failed to update default user settings";
      const isServerMessage = ["server", "string", "envelope"].includes(result);
      expect(await screen.findByText(isServerMessage ? "Home" : translate("zh-CN", message))).toBeVisible();
      if (["server", "envelope"].includes(result))
        expect(screen.getByText(translate("zh-CN", "Access Denied"))).toBeVisible();
      expect(screen.queryByText("首页")).not.toBeInTheDocument();
      fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
      expect(await screen.findByText(isServerMessage ? "Home" : message)).toBeVisible();
      expect(updateSettings).toHaveBeenCalledTimes(1);
      expect(updateSettings).toHaveBeenCalledWith({ ...SAVED_BODY, max_budget: 250 });
      if (result !== "success") expect(screen.getByLabelText("Max Budget (USD)")).toHaveValue(250);
    },
  );

  it("shows a read-only summary until Edit Settings is clicked", async () => {
    renderForm();

    expect(await screen.findByText("Internal User")).toBeInTheDocument();
    expect(screen.getByText("100")).toBeInTheDocument();
    expect(screen.getByText("monthly")).toBeInTheDocument();
    expect(screen.getByText("gpt-5.2")).toBeInTheDocument();
    expect(screen.getByText(/team-alpha/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save Changes" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Max Budget (USD)")).not.toBeInTheDocument();
  });

  it("labels model sentinels in the read-only summary", async () => {
    renderForm({
      fetchSettings: vi
        .fn()
        .mockResolvedValue({ ...SETTINGS, values: { ...SETTINGS.values, models: ["all-proxy-models"] } }),
    });

    expect(await screen.findByText("All Proxy Models")).toBeInTheDocument();
  });

  it("disables Save until the loaded settings are edited", async () => {
    const user = userEvent.setup();
    renderForm();

    await enterEditMode(user);

    expect(await saveButton()).toBeDisabled();
  });

  it("shows an error instead of the form when the settings cannot be loaded", async () => {
    renderForm({ fetchSettings: vi.fn().mockRejectedValue(new Error("nope")) });

    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load the default user settings.");
    expect(screen.queryByRole("button", { name: "Edit Settings" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save Changes" })).not.toBeInTheDocument();
  });

  it("sends every field on save, not only the edited one", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await enterEditMode(user);
    await user.clear(await screen.findByLabelText("Max Budget (USD)"));
    await user.type(screen.getByLabelText("Max Budget (USD)"), "250");
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({ ...SAVED_BODY, max_budget: 250 });
  });

  it("saves a sub-cent budget the browser would veto under a 0.01 step", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await enterEditMode(user);
    const budget: HTMLInputElement = await screen.findByLabelText("Max Budget (USD)");
    await user.clear(budget);
    await user.type(budget, "0.001");

    const teamBudget: HTMLInputElement = screen.getByLabelText("Max Budget in Team (USD)");
    await user.clear(teamBudget);
    await user.type(teamBudget, "0.002");

    // jsdom never blocks the submit itself, so assert the constraint the real browser
    // enforces before handleSubmit ever runs
    expect(budget.checkValidity()).toBe(true);
    expect(teamBudget.checkValidity()).toBe(true);

    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({
      ...SAVED_BODY,
      max_budget: 0.001,
      teams: [{ team_id: "team-alpha", max_budget_in_team: 0.002, user_role: "user" }],
    });
  });

  it("clears an emptied budget with null", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await enterEditMode(user);
    await user.clear(await screen.findByLabelText("Max Budget (USD)"));
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({ ...SAVED_BODY, max_budget: null });
  });

  it("sends the models selection through unchanged, sentinel values included", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await enterEditMode(user);
    await user.click(await screen.findByRole("button", { name: "set-models" }));
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({ ...SAVED_BODY, models: ["all-proxy-models"] });
  });

  it("saves a team that was picked from the searchable list", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await enterEditMode(user);
    await user.click(await screen.findByRole("button", { name: "Add Team" }));
    await user.click(screen.getAllByLabelText("Team")[1]);
    await user.click(await screen.findByText("Beta"));
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({
      ...SAVED_BODY,
      teams: [
        { team_id: "team-alpha", max_budget_in_team: 25, user_role: "user" },
        { team_id: "team-beta", max_budget_in_team: null, user_role: "user" },
      ],
    });
  });

  it("never turns a team id typed into the picker into a saved team", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await enterEditMode(user);
    await user.click(await screen.findByRole("button", { name: "Add Team" }));
    await user.type(screen.getAllByLabelText("Team")[1], "team-alhpa");
    await user.keyboard("{Escape}");
    await user.click(await saveButton());

    expect(await screen.findByText("Select a team")).toBeInTheDocument();
    expect(screen.getAllByLabelText("Team")[1]).toHaveValue("");
    expect(updateSettings).not.toHaveBeenCalled();
  });

  it("blocks saving the same default team twice", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await enterEditMode(user);
    await user.click(await screen.findByRole("button", { name: "Add Team" }));
    await user.click(screen.getAllByLabelText("Team")[1]);
    await user.click(await screen.findByText("Alpha"));
    await user.click(await saveButton());

    expect(await screen.findByText("This team is already listed")).toBeInTheDocument();
    expect(updateSettings).not.toHaveBeenCalled();
  });

  it("drops a removed team row from the saved settings", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await enterEditMode(user);
    await user.click(await screen.findByRole("button", { name: "Remove" }));
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(updateSettings).toHaveBeenCalledWith({ ...SAVED_BODY, teams: null });
  });

  it("returns to the read-only view showing the new values after a successful save", async () => {
    const user = userEvent.setup();
    const updated = { ...SETTINGS, values: { ...SETTINGS.values, max_budget: 250 } };
    const { updateSettings } = renderForm({
      fetchSettings: vi.fn().mockResolvedValueOnce(SETTINGS).mockResolvedValue(updated),
    });

    await enterEditMode(user);
    await user.clear(await screen.findByLabelText("Max Budget (USD)"));
    await user.type(screen.getByLabelText("Max Budget (USD)"), "250");
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(await screen.findByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    expect(await screen.findByText("250")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save Changes" })).not.toBeInTheDocument();
    expect(await screen.findByText("Default user settings updated successfully")).toBeVisible();

    await enterEditMode(user);
    expect(await saveButton()).toBeDisabled();
  });

  it("keeps the edit and surfaces the backend error when the save fails", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm({
      updateSettings: vi.fn().mockRejectedValue(new Error("Team(s) not found: team-alhpa.")),
    });

    await enterEditMode(user);
    await user.clear(await screen.findByLabelText("Max Budget (USD)"));
    await user.type(screen.getByLabelText("Max Budget (USD)"), "250");
    await user.click(await saveButton());

    await waitFor(() => expect(updateSettings).toHaveBeenCalledTimes(1));
    expect(await screen.findByText("Team(s) not found: team-alhpa.")).toBeVisible();
    expect(await saveButton()).toBeEnabled();
    expect(screen.getByLabelText("Max Budget (USD)")).toHaveValue(250);
  });

  it("discards edits and returns to the read-only view when Cancel is pressed", async () => {
    const user = userEvent.setup();
    const { updateSettings } = renderForm();

    await enterEditMode(user);
    await user.clear(await screen.findByLabelText("Max Budget (USD)"));
    await user.type(screen.getByLabelText("Max Budget (USD)"), "250");
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(await screen.findByRole("button", { name: "Edit Settings" })).toBeInTheDocument();
    expect(screen.getByText("100")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Save Changes" })).not.toBeInTheDocument();
    expect(updateSettings).not.toHaveBeenCalled();

    await enterEditMode(user);
    expect(screen.getByLabelText("Max Budget (USD)")).toHaveValue(100);
    expect(await saveButton()).toBeDisabled();
  });
});
