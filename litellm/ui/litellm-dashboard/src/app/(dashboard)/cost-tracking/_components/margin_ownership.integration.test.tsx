import React from "react";
import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../../../tests/test-utils";
import CostTrackingSettings from "./cost_tracking_settings";

describe("Margin configuration ownership", () => {
  it.each(["config", "db", "unset"])("only locks margins controlled by config.yaml, with source %s", async (source) => {
    vi.spyOn(global, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      const responses = {
        "/config/cost_margin_config": { values: { openai: 0.1 }, source, editable: source !== "config" },
        "/config/cost_discount_config": { values: {} },
        "/config/block_requests_for_models_without_pricing": { enabled: false },
      };
      const body = Object.entries(responses).find(([endpoint]) => url.includes(endpoint))?.[1] ?? { data: [] };
      return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
    });
    const user = userEvent.setup();
    renderWithProviders(<CostTrackingSettings userID="admin" userRole="proxy_admin" accessToken="test-token" />);
    await user.click(screen.getByRole("button", { name: /Fee\/Price Margin/ }));
    const add = await screen.findByRole("button", { name: /add provider margin/i });
    const edit = await screen.findByRole("button", { name: /edit margin for openai/i });
    const remove = screen.getByRole("button", { name: /remove margin for openai/i });

    if (source === "config") {
      expect(screen.getByText("config.yaml")).toBeInTheDocument();
      expect(add).toBeDisabled();
      expect(edit).toBeDisabled();
      expect(remove).toBeDisabled();
    } else {
      expect(screen.queryByText("config.yaml")).not.toBeInTheDocument();
      expect(add).toBeEnabled();
      expect(edit).toBeEnabled();
      expect(remove).toBeEnabled();
    }
  });
});
