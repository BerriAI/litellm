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

describe("LoggingSettingsView scoped exporters", () => {
  it("says nothing about exporters when the caller has not resolved them", () => {
    render(<LoggingSettingsView loggingConfigs={[]} disabledCallbacks={[]} />);

    expect(screen.queryByText("No logging exporters assigned")).not.toBeInTheDocument();
    expect(screen.queryByText("Logging Exporters")).not.toBeInTheDocument();
  });

  it("reports none only when the caller resolved an empty list", () => {
    render(<LoggingSettingsView loggingConfigs={[]} disabledCallbacks={[]} scopedExporters={[]} />);

    expect(screen.getByText("No logging exporters assigned")).toBeInTheDocument();
  });

  it("lists the exporters the caller resolved", () => {
    render(<LoggingSettingsView loggingConfigs={[]} disabledCallbacks={[]} scopedExporters={["langfuse-eu"]} />);

    expect(screen.getByText("langfuse-eu")).toBeInTheDocument();
    expect(screen.queryByText("No logging exporters assigned")).not.toBeInTheDocument();
  });
});
