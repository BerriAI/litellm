import { describe, it, expect, beforeAll } from "vitest";
import {
  client,
  modelListV1ModelsGet,
  healthEndpointHealthGet,
  modelInfoV1ModelInfoGet,
} from "./index";

const PROXY_URL = "http://localhost:4000";
const API_KEY = "sk-1234";

describe("E2E SDK tests against live proxy", () => {
  beforeAll(() => {
    client.setConfig({
      baseUrl: PROXY_URL,
      headers: {
        Authorization: `Bearer ${API_KEY}`,
      },
    });
  });

  it("should call /health endpoint", async () => {
    const result = await healthEndpointHealthGet();
    expect(result.data).toBeDefined();
  }, 30_000);

  it("should list models via /v1/models", async () => {
    const result = await modelListV1ModelsGet();
    expect(result.data).toBeDefined();
  });

  it("should get model info via /v1/model/info", async () => {
    const result = await modelInfoV1ModelInfoGet();
    expect(result.data).toBeDefined();
  });
});
