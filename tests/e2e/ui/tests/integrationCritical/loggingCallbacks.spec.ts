import { expect, test } from "@playwright/test";
import { dismissFeedbackPopup } from "../../helpers/navigation";
import { masterKey } from "../../helpers/traffic";

test("switching callback providers drops the previous credentials and a rejected save keeps the current draft", async ({
  page,
}) => {
  const failure = "Callback could not be saved";
  const writes: unknown[] = [];
  await page.route("**/config/update", async (route) => {
    if (route.request().method() !== "POST") {
      await route.continue();
      return;
    }
    writes.push(route.request().postDataJSON());
    expect.soft(writes, "Only one callback write is expected").toHaveLength(1);
    await route.fulfill({ status: 500, json: { detail: failure } });
  });

  await page.goto("/ui/login");
  await page.getByPlaceholder("Enter your username").fill("admin");
  await page.getByPlaceholder("Enter your password").fill(masterKey());
  await page.getByRole("button", { name: "Login", exact: true }).click();
  await expect(page).toHaveURL(
    (url) => url.pathname.startsWith("/ui") && !url.pathname.includes("/login"),
  );
  await page.goto("/ui/logging-and-alerts");
  await dismissFeedbackPopup(page);

  const addCallback = page.getByRole("button", {
    name: "Add Callback",
    exact: true,
  });
  const dialog = page.getByRole("dialog", {
    name: "Add Logging Callback",
    exact: true,
  });
  const selectProvider = async (name: RegExp) => {
    await dialog
      .getByRole("combobox", { name: "Callback", exact: true })
      .click();
    await page.getByRole("option", { name }).click();
  };
  await addCallback.click();
  await selectProvider(/Langfuse$/);
  await dialog
    .getByLabel("Public Key", { exact: true })
    .fill("e2e-placeholder-public");
  await dialog
    .getByLabel("Secret Key", { exact: true })
    .fill("e2e-placeholder-secret");
  await selectProvider(/Open Telemetry$/);
  const endpoint = dialog.getByLabel("Endpoint URL", { exact: true });
  const draftUrl = "https://example.invalid/telemetry";
  await endpoint.fill(draftUrl);
  const protocol = dialog.getByRole("combobox", {
    name: "Export Protocol",
    exact: true,
  });
  await protocol.click();
  await page.getByRole("option", { name: "http/json", exact: true }).click();
  const backgroundSpans = dialog.getByRole("switch", {
    name: "Emit Background Service Spans",
    exact: true,
  });
  await backgroundSpans.click();

  await dialog
    .getByRole("button", { name: "Add Callback", exact: true })
    .click();
  await expect(page.getByText(failure, { exact: true })).toBeVisible();
  await expect(endpoint).toHaveValue(draftUrl);
  await expect(protocol.getByText("http/json", { exact: true })).toBeVisible();
  await expect(backgroundSpans).toHaveAttribute("aria-checked", "false");
  expect(writes).toEqual([
    {
      environment_variables: {
        callback: "otel",
        otel_endpoint: draftUrl,
        otel_traces_endpoint: "",
        otel_headers: "",
        otel_exporter_otlp_protocol: "http/json",
        LITELLM_OTEL_EMIT_BACKGROUND_SERVICE_SPANS: "false",
      },
      litellm_settings: { success_callback: ["otel"] },
    },
  ]);
});
