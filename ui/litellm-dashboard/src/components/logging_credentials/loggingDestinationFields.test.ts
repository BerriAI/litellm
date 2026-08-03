import { describe, expect, it } from "vitest";

import { LOGGING_DESTINATION_BACKENDS } from "./loggingDestinationFields";

// Arize routes a trace to a project via the model_id / arize.project.name span
// resource attribute, and its OTLP ingestion rejects any span that lacks it
// ("model_id span resource attribute or arize.project.name span attribute is
// required"). The backend reads that project from the credential's
// arize_project_name value, so the create form must collect it as a required
// field; without it every Arize destination created in the UI silently drops
// 100% of its traces.
describe("Arize logging destination fields", () => {
  const arize = LOGGING_DESTINATION_BACKENDS.find((b) => b.id === "arize");

  it("exposes a required arize_project_name field", () => {
    expect(arize).toBeDefined();
    const projectField = arize!.fields.find((f) => f.name === "arize_project_name");
    expect(projectField).toBeDefined();
    expect(projectField!.optional).not.toBe(true);
  });
});
