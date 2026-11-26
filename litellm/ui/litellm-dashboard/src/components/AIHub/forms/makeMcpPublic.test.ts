import { describe, expect, it } from "vitest";
import { toMcpPublicationRequest } from "./makeMcpPublicMapper";
import { makeMcpPublicSchema } from "./makeMcpPublicSchema";

describe("Mcp publication", () => {
  it("keeps the selected identifiers in selection order in the request", () => {
    const serverIds = ["server-z", "server-a"];
    expect(toMcpPublicationRequest(makeMcpPublicSchema.parse({ serverIds }))).toEqual({
      mcp_server_ids: serverIds,
    });
  });

  it("sends an empty allowlist when nothing is selected", () => {
    expect(toMcpPublicationRequest(makeMcpPublicSchema.parse({ serverIds: [] }))).toEqual({ mcp_server_ids: [] });
  });

  it("rejects missing selections rather than publishing an empty allowlist implicitly", () => {
    expect(makeMcpPublicSchema.safeParse({}).success).toBe(false);
  });
});
