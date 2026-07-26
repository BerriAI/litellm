import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LoggingSettingsView } from "./logging_settings_view";

describe("LoggingSettingsView logos", () => {
  it("renders the bundled logo for a known logging integration", () => {
    render(
      <LoggingSettingsView
        loggingConfigs={[{ callback_name: "langfuse", callback_type: "success", callback_vars: {} }]}
      />,
    );

    expect(screen.getByAltText("Langfuse logo")).toHaveAttribute("src", "/_next/static/media/langfuse.png");
  });

  it("renders the bundled logo for a disabled callback given by internal slug", () => {
    render(<LoggingSettingsView disabledCallbacks={["datadog"]} />);

    expect(screen.getByAltText("Datadog logo")).toHaveAttribute("src", "/_next/static/media/datadog.png");
  });

  it("renders a letter avatar for an unknown callback name", () => {
    render(
      <LoggingSettingsView
        loggingConfigs={[{ callback_name: "mystery_callback", callback_type: "success", callback_vars: {} }]}
      />,
    );

    expect(document.querySelector("img")).toBeNull();
    expect(screen.getByText("m")).toBeInTheDocument();
    expect(screen.getByText("mystery_callback")).toBeInTheDocument();
  });

  it("renders a letter avatar for the custom callback API, which has no bundled logo", () => {
    render(<LoggingSettingsView disabledCallbacks={["custom_callback_api"]} />);

    expect(screen.queryByAltText("Custom Callback API logo")).toBeNull();
    expect(screen.getByText("C")).toBeInTheDocument();
  });
});
