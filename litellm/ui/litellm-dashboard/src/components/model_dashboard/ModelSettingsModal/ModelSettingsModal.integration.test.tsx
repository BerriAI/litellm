import { toast } from "@/lib/toast";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { jsonResponse, renderWithProviders, screen, waitFor } from "../../../../tests/test-utils";
import ModelSettingsModal from "./ModelSettingsModal";

it("saves the stored string value unchanged when the switch is left alone", async () => {
  const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (url) => {
    if (String(url).includes("/config/list")) {
      return jsonResponse([{ field_name: "store_model_in_db", field_value: "True" }]);
    }
    return jsonResponse({ message: "Settings saved" });
  });

  try {
    renderWithProviders(<ModelSettingsModal isVisible onCancel={vi.fn()} />);
    expect(await screen.findByRole("switch", { checked: true })).toBeInTheDocument();
    await userEvent.setup().click(screen.getByRole("button", { name: "Save Settings" }));

    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("Model storage settings updated successfully"));
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringMatching(/\/config\/field\/update$/),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          field_name: "store_model_in_db",
          field_value: "True",
          config_type: "general_settings",
        }),
      }),
    );
  } finally {
    fetchSpy.mockRestore();
  }
});
