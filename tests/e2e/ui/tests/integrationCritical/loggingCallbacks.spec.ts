import { expect, test } from "@playwright/test";
import { dismissFeedbackPopup } from "../../helpers/navigation";
import { masterKey } from "../../helpers/traffic";

test("switching callback providers drops the previous credentials and a rejected save keeps the current draft", async ({
  page,
}) => {
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
  await dialog
    .getByRole("combobox", { name: "Export Protocol", exact: true })
    .click();
  await page.getByRole("option", { name: "http/json", exact: true }).click();
  await dialog
    .getByRole("switch", { name: "Emit Background Service Spans", exact: true })
    .click();

  const failure = "Callback could not be saved";
  await page.route(
    "**/config/update",
    (route) => route.fulfill({ status: 500, json: { detail: failure } }),
    {
      times: 1,
    },
  );
  const write = page.waitForRequest(
    (request) =>
      request.method() === "POST" &&
      new URL(request.url()).pathname === "/config/update",
  );
  await dialog
    .getByRole("button", { name: "Add Callback", exact: true })
    .click();
  expect((await write).postDataJSON()).toEqual({
    environment_variables: {
      callback: "otel",
      otel_endpoint: draftUrl,
      otel_traces_endpoint: "",
      otel_headers: "",
      otel_exporter_otlp_protocol: "http/json",
      LITELLM_OTEL_EMIT_BACKGROUND_SERVICE_SPANS: "false",
    },
    litellm_settings: { success_callback: ["otel"] },
  });
  await expect(page.getByText(failure, { exact: true })).toBeVisible();
  await expect(endpoint).toHaveValue(draftUrl);
});
