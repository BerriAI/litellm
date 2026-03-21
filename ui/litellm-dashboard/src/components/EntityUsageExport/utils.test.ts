import type { DateRangePickerValue } from "@tremor/react";
import Papa from "papaparse";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { EntitySpendData, ExportScope } from "./types";
import {
  generateDailyData,
  generateDailyWithKeysData,
  generateDailyWithModelsData,
  generateExportData,
  generateMetadata,
  getEntityBreakdown,
  handleExportCSV,
  handleExportJSON,
} from "./utils";

vi.mock("@/utils/dataUtils", () => ({
  formatNumberWithCommas: vi.fn((value: number, decimals: number = 0) => {
    if (value === null || value === undefined || !Number.isFinite(value)) {
      return "-";
    }
    return value.toFixed(decimals);
  }),
}));

vi.mock("papaparse", () => ({
  default: {
    unparse: vi.fn((data: any[]) => "mocked-csv-data"),
  },
}));

describe("EntityUsageExport utils", () => {
  const mockSpendData: EntitySpendData = {
    results: [
      {
        date: "2025-01-01",
        breakdown: {
          entities: {
            entity1: {
              metrics: {
                spend: 10.5,
                api_requests: 100,
                successful_requests: 95,
                failed_requests: 5,
                total_tokens: 1000,
                prompt_tokens: 600,
                completion_tokens: 400,
                cache_read_input_tokens: 50,
                cache_creation_input_tokens: 30,
              },
              api_key_breakdown: {
                key1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                  },
                  metadata: {
                    team_id: "team-1",
                    key_alias: "alias-1",
                  },
                },
              },
            },
            entity2: {
              metrics: {
                spend: 20.3,
                api_requests: 200,
                successful_requests: 190,
                failed_requests: 10,
                total_tokens: 2000,
                prompt_tokens: 1200,
                completion_tokens: 800,
                cache_read_input_tokens: 100,
                cache_creation_input_tokens: 60,
              },
              api_key_breakdown: {
                key2: {
                  metrics: {
                    spend: 20.3,
                    api_requests: 200,
                    successful_requests: 190,
                    failed_requests: 10,
                    total_tokens: 2000,
                  },
                  metadata: {
                    team_id: "team-2",
                    key_alias: "alias-2",
                  },
                },
              },
            },
          },
        },
      },
      {
        date: "2025-01-02",
        breakdown: {
          entities: {
            entity1: {
              metrics: {
                spend: 15.2,
                api_requests: 150,
                successful_requests: 145,
                failed_requests: 5,
                total_tokens: 1500,
                prompt_tokens: 900,
                completion_tokens: 600,
                cache_read_input_tokens: 75,
                cache_creation_input_tokens: 45,
              },
              api_key_breakdown: {
                key1: {
                  metrics: {
                    spend: 15.2,
                    api_requests: 150,
                    successful_requests: 145,
                    failed_requests: 5,
                    total_tokens: 1500,
                  },
                  metadata: {
                    team_id: "team-1",
                    key_alias: "alias-1",
                  },
                },
              },
            },
          },
        },
      },
    ],
    metadata: {
      total_spend: 46.0,
      total_api_requests: 450,
      total_successful_requests: 430,
      total_failed_requests: 20,
      total_tokens: 4500,
    },
  };

  const mockTeamAliasMap: Record<string, string> = {
    "team-1": "Team One",
    "team-2": "Team Two",
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("getEntityBreakdown", () => {
    it("should aggregate entity spend data across multiple days", () => {
      const result = getEntityBreakdown(mockSpendData);

      expect(result).toHaveLength(2);
      expect(result[0].metadata.id).toBe("team-1");
      expect(result[0].metrics.spend).toBe(25.7);
      expect(result[1].metadata.id).toBe("team-2");
      expect(result[1].metrics.spend).toBe(20.3);
    });

    it("should sort entities by spend descending", () => {
      const result = getEntityBreakdown(mockSpendData);

      expect(result[0].metrics.spend).toBeGreaterThan(result[1].metrics.spend);
    });

    it("should aggregate all metrics correctly", () => {
      const result = getEntityBreakdown(mockSpendData);
      const entity1 = result.find((e) => e.metadata.id === "team-1");

      expect(entity1?.metrics.api_requests).toBe(250);
      expect(entity1?.metrics.successful_requests).toBe(240);
      expect(entity1?.metrics.failed_requests).toBe(10);
      expect(entity1?.metrics.total_tokens).toBe(2500);
      expect(entity1?.metrics.prompt_tokens).toBe(1500);
      expect(entity1?.metrics.completion_tokens).toBe(1000);
      expect(entity1?.metrics.cache_read_input_tokens).toBe(125);
      expect(entity1?.metrics.cache_creation_input_tokens).toBe(75);
    });

    it("should use key alias when available", () => {
      const result = getEntityBreakdown(mockSpendData);
      const entity1 = result.find((e) => e.metadata.id === "team-1");

      expect(entity1?.metadata.alias).toBe("alias-1");
    });

    it("should use team alias map when key alias is not available", () => {
      const spendDataWithoutAlias: EntitySpendData = {
        ...mockSpendData,
        results: [
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                entity1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                    prompt_tokens: 600,
                    completion_tokens: 400,
                    cache_read_input_tokens: 50,
                    cache_creation_input_tokens: 30,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 10.5,
                        api_requests: 100,
                        successful_requests: 95,
                        failed_requests: 5,
                        total_tokens: 1000,
                      },
                      metadata: {
                        team_id: "team-1",
                      },
                    },
                  },
                },
              },
            },
          },
        ],
        metadata: mockSpendData.metadata,
      };

      const result = getEntityBreakdown(spendDataWithoutAlias, mockTeamAliasMap);
      const entity1 = result.find((e) => e.metadata.id === "team-1");

      expect(entity1?.metadata.alias).toBe("Team One");
    });

    it("should use entity id when team alias is not available", () => {
      const spendDataWithoutTeamId: EntitySpendData = {
        ...mockSpendData,
        results: [
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                entity1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                    prompt_tokens: 600,
                    completion_tokens: 400,
                    cache_read_input_tokens: 50,
                    cache_creation_input_tokens: 30,
                  },
                  api_key_breakdown: {},
                },
              },
            },
          },
        ],
        metadata: mockSpendData.metadata,
      };

      const result = getEntityBreakdown(spendDataWithoutTeamId);
      const entity1 = result.find((e) => e.metadata.id === "entity1");

      expect(entity1?.metadata.alias).toBe("entity1");
    });

    it("should handle empty spend data", () => {
      const emptySpendData: EntitySpendData = {
        results: [],
        metadata: {
          total_spend: 0,
          total_api_requests: 0,
          total_successful_requests: 0,
          total_failed_requests: 0,
          total_tokens: 0,
        },
      };

      const result = getEntityBreakdown(emptySpendData);

      expect(result).toHaveLength(0);
    });

    it("should handle missing optional token fields", () => {
      const spendDataWithMissingTokens: EntitySpendData = {
        ...mockSpendData,
        results: [
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                entity1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 10.5,
                        api_requests: 100,
                        successful_requests: 95,
                        failed_requests: 5,
                        total_tokens: 1000,
                      },
                      metadata: {
                        team_id: "team-1",
                      },
                    },
                  },
                },
              },
            },
          },
        ],
        metadata: mockSpendData.metadata,
      };

      const result = getEntityBreakdown(spendDataWithMissingTokens);
      const entity1 = result.find((e) => e.metadata.id === "team-1");

      expect(entity1?.metrics.prompt_tokens).toBe(0);
      expect(entity1?.metrics.completion_tokens).toBe(0);
    });
  });

  describe("generateDailyData", () => {
    it("should generate daily breakdown data with correct structure", () => {
      const result = generateDailyData(mockSpendData, "Team", mockTeamAliasMap);

      expect(result).toHaveLength(3);
      expect(result[0]).toHaveProperty("Date");
      expect(result[0]).toHaveProperty("Team");
      expect(result[0]).toHaveProperty("Team ID");
      expect(result[0]).toHaveProperty("Spend ($)");
      expect(result[0]).toHaveProperty("Requests");
      expect(result[0]).toHaveProperty("Successful Requests");
      expect(result[0]).toHaveProperty("Failed Requests");
      expect(result[0]).toHaveProperty("Total Tokens");
      expect(result[0]).toHaveProperty("Prompt Tokens");
      expect(result[0]).toHaveProperty("Completion Tokens");
    });

    it("should sort data by date ascending", () => {
      const result = generateDailyData(mockSpendData, "Team");

      const dates = result.map((r) => new Date(r.Date).getTime());
      for (let i = 0; i < dates.length - 1; i++) {
        expect(dates[i]).toBeLessThanOrEqual(dates[i + 1]);
      }
    });

    it("should use team alias when available", () => {
      const result = generateDailyData(mockSpendData, "Team", mockTeamAliasMap);
      const team1Entry = result.find((r) => r["Team ID"] === "team-1");

      expect(team1Entry?.["Team"]).toBe("Team One");
    });

    it("should use dash when team alias is not available", () => {
      const result = generateDailyData(mockSpendData, "Team");
      const entryWithoutTeamId = result.find((r) => !r["Team ID"] || r["Team ID"] === "-");

      if (entryWithoutTeamId) {
        expect(entryWithoutTeamId["Team"]).toBe("-");
      }
    });

    it("should use dash when team id is not available", () => {
      const spendDataWithoutTeamId: EntitySpendData = {
        ...mockSpendData,
        results: [
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                entity1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                    prompt_tokens: 600,
                    completion_tokens: 400,
                  },
                  api_key_breakdown: {},
                },
              },
            },
          },
        ],
        metadata: mockSpendData.metadata,
      };

      const result = generateDailyData(spendDataWithoutTeamId, "Team");
      const entry = result[0];

      expect(entry["Team ID"]).toBe("-");
      expect(entry["Team"]).toBe("-");
    });

    it("should format spend values correctly", () => {
      const result = generateDailyData(mockSpendData, "Team");

      expect(result[0]["Spend ($)"]).toBeDefined();
    });

    it("should handle missing optional token fields", () => {
      const spendDataWithMissingTokens: EntitySpendData = {
        ...mockSpendData,
        results: [
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                entity1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 10.5,
                        api_requests: 100,
                        successful_requests: 95,
                        failed_requests: 5,
                        total_tokens: 1000,
                      },
                      metadata: {
                        team_id: "team-1",
                      },
                    },
                  },
                },
              },
            },
          },
        ],
        metadata: mockSpendData.metadata,
      };

      const result = generateDailyData(spendDataWithMissingTokens, "Team");

      expect(result[0]["Prompt Tokens"]).toBe(0);
      expect(result[0]["Completion Tokens"]).toBe(0);
    });
  });

  describe("generateDailyWithKeysData", () => {
    const mockSpendDataWithKeys: EntitySpendData = {
      results: [
        {
          date: "2025-01-01",
          breakdown: {
            entities: {
              entity1: {
                metrics: {
                  spend: 10.5,
                  api_requests: 100,
                  successful_requests: 95,
                  failed_requests: 5,
                  total_tokens: 1000,
                  prompt_tokens: 600,
                  completion_tokens: 400,
                },
                api_key_breakdown: {
                  key1: {
                    metrics: {
                      spend: 5.0,
                      api_requests: 50,
                      successful_requests: 48,
                      failed_requests: 2,
                      total_tokens: 500,
                      prompt_tokens: 300,
                      completion_tokens: 200,
                    },
                    metadata: {
                      team_id: "team-1",
                      key_alias: "alias-1",
                    },
                  },
                  key2: {
                    metrics: {
                      spend: 5.5,
                      api_requests: 50,
                      successful_requests: 47,
                      failed_requests: 3,
                      total_tokens: 500,
                      prompt_tokens: 300,
                      completion_tokens: 200,
                    },
                    metadata: {
                      team_id: "team-1",
                      key_alias: "alias-2",
                    },
                  },
                },
              },
              entity2: {
                metrics: {
                  spend: 20.3,
                  api_requests: 200,
                  successful_requests: 190,
                  failed_requests: 10,
                  total_tokens: 2000,
                  prompt_tokens: 1200,
                  completion_tokens: 800,
                },
                api_key_breakdown: {
                  key3: {
                    metrics: {
                      spend: 20.3,
                      api_requests: 200,
                      successful_requests: 190,
                      failed_requests: 10,
                      total_tokens: 2000,
                      prompt_tokens: 1200,
                      completion_tokens: 800,
                    },
                    metadata: {
                      team_id: "team-2",
                      key_alias: "alias-3",
                    },
                  },
                },
              },
            },
          },
        },
        {
          date: "2025-01-02",
          breakdown: {
            entities: {
              entity1: {
                metrics: {
                  spend: 15.2,
                  api_requests: 150,
                  successful_requests: 145,
                  failed_requests: 5,
                  total_tokens: 1500,
                  prompt_tokens: 900,
                  completion_tokens: 600,
                },
                api_key_breakdown: {
                  key1: {
                    metrics: {
                      spend: 15.2,
                      api_requests: 150,
                      successful_requests: 145,
                      failed_requests: 5,
                      total_tokens: 1500,
                      prompt_tokens: 900,
                      completion_tokens: 600,
                    },
                    metadata: {
                      team_id: "team-1",
                      key_alias: "alias-1",
                    },
                  },
                },
              },
            },
          },
        },
      ],
      metadata: {
        total_spend: 46.0,
        total_api_requests: 450,
        total_successful_requests: 430,
        total_failed_requests: 20,
        total_tokens: 4500,
      },
    };

    it("should generate daily breakdown with key data and correct structure", () => {
      const result = generateDailyWithKeysData(mockSpendDataWithKeys, "Team", mockTeamAliasMap);

      expect(result.length).toBeGreaterThan(0);
      expect(result[0]).toHaveProperty("Date");
      expect(result[0]).toHaveProperty("Team");
      expect(result[0]).toHaveProperty("Team ID");
      expect(result[0]).toHaveProperty("Key Alias");
      expect(result[0]).toHaveProperty("Key ID");
      expect(result[0]).toHaveProperty("Spend ($)");
      expect(result[0]).toHaveProperty("Requests");
      expect(result[0]).toHaveProperty("Successful Requests");
      expect(result[0]).toHaveProperty("Failed Requests");
      expect(result[0]).toHaveProperty("Total Tokens");
      expect(result[0]).toHaveProperty("Prompt Tokens");
      expect(result[0]).toHaveProperty("Completion Tokens");
    });

    it("should sort data by date ascending", () => {
      const result = generateDailyWithKeysData(mockSpendDataWithKeys, "Team");

      const dates = result.map((r) => new Date(r.Date).getTime());
      for (let i = 0; i < dates.length - 1; i++) {
        expect(dates[i]).toBeLessThanOrEqual(dates[i + 1]);
      }
    });

    it("should aggregate metrics for duplicate date-team-key combinations", () => {
      const spendDataWithDuplicates: EntitySpendData = {
        results: [
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                entity1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                    prompt_tokens: 600,
                    completion_tokens: 400,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 5.0,
                        api_requests: 50,
                        successful_requests: 48,
                        failed_requests: 2,
                        total_tokens: 500,
                        prompt_tokens: 300,
                        completion_tokens: 200,
                      },
                      metadata: {
                        team_id: "team-1",
                        key_alias: "alias-1",
                      },
                    },
                  },
                },
              },
            },
          },
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                entity1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                    prompt_tokens: 600,
                    completion_tokens: 400,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 5.0,
                        api_requests: 50,
                        successful_requests: 47,
                        failed_requests: 3,
                        total_tokens: 500,
                        prompt_tokens: 300,
                        completion_tokens: 200,
                      },
                      metadata: {
                        team_id: "team-1",
                        key_alias: "alias-1",
                      },
                    },
                  },
                },
              },
            },
          },
        ],
        metadata: {
          total_spend: 21.0,
          total_api_requests: 200,
          total_successful_requests: 190,
          total_failed_requests: 10,
          total_tokens: 2000,
        },
      };

      const result = generateDailyWithKeysData(spendDataWithDuplicates, "Team");
      const key1Entries = result.filter((r) => r["Key ID"] === "key1");

      expect(key1Entries).toHaveLength(1);
      expect(key1Entries[0].Requests).toBe(100);
      expect(key1Entries[0]["Successful Requests"]).toBe(95);
      expect(key1Entries[0]["Failed Requests"]).toBe(5);
      expect(key1Entries[0]["Total Tokens"]).toBe(1000);
    });

    it("should use team alias when available", () => {
      const result = generateDailyWithKeysData(mockSpendDataWithKeys, "Team", mockTeamAliasMap);
      const team1Entry = result.find((r) => r["Team ID"] === "team-1");

      expect(team1Entry?.["Team"]).toBe("Team One");
    });

    it("should use dash when team alias is not available", () => {
      const result = generateDailyWithKeysData(mockSpendDataWithKeys, "Team");
      const entryWithoutTeamAlias = result.find((r) => r["Team ID"] === "team-1" && !mockTeamAliasMap[r["Team ID"]]);

      if (entryWithoutTeamAlias) {
        expect(entryWithoutTeamAlias["Team"]).toBe("-");
      }
    });

    it("should use key alias when available", () => {
      const result = generateDailyWithKeysData(mockSpendDataWithKeys, "Team");
      const key1Entry = result.find((r) => r["Key ID"] === "key1");

      expect(key1Entry?.["Key Alias"]).toBe("alias-1");
    });

    it("should use dash when key alias is not available", () => {
      const spendDataWithoutKeyAlias: EntitySpendData = {
        ...mockSpendDataWithKeys,
        results: [
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                entity1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                    prompt_tokens: 600,
                    completion_tokens: 400,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 10.5,
                        api_requests: 100,
                        successful_requests: 95,
                        failed_requests: 5,
                        total_tokens: 1000,
                        prompt_tokens: 600,
                        completion_tokens: 400,
                      },
                      metadata: {
                        team_id: "team-1",
                      },
                    },
                  },
                },
              },
            },
          },
        ],
        metadata: mockSpendDataWithKeys.metadata,
      };

      const result = generateDailyWithKeysData(spendDataWithoutKeyAlias, "Team");
      const key1Entry = result.find((r) => r["Key ID"] === "key1");

      expect(key1Entry?.["Key Alias"]).toBe("-");
    });

    it("should use entity id when team id is not available in metadata", () => {
      const spendDataWithoutTeamId: EntitySpendData = {
        ...mockSpendDataWithKeys,
        results: [
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                entity1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                    prompt_tokens: 600,
                    completion_tokens: 400,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 10.5,
                        api_requests: 100,
                        successful_requests: 95,
                        failed_requests: 5,
                        total_tokens: 1000,
                        prompt_tokens: 600,
                        completion_tokens: 400,
                      },
                      metadata: {},
                    },
                  },
                },
              },
            },
          },
        ],
        metadata: mockSpendDataWithKeys.metadata,
      };

      const result = generateDailyWithKeysData(spendDataWithoutTeamId, "Team");
      const entry = result.find((r) => r["Key ID"] === "key1");

      expect(entry?.["Team ID"]).toBe("entity1");
    });

    it("should use dash when team id is not available", () => {
      const spendDataWithoutTeamId: EntitySpendData = {
        ...mockSpendDataWithKeys,
        results: [
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                entity1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                    prompt_tokens: 600,
                    completion_tokens: 400,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 10.5,
                        api_requests: 100,
                        successful_requests: 95,
                        failed_requests: 5,
                        total_tokens: 1000,
                        prompt_tokens: 600,
                        completion_tokens: 400,
                      },
                      metadata: {
                        team_id: null,
                      },
                    },
                  },
                },
              },
            },
          },
        ],
        metadata: mockSpendDataWithKeys.metadata,
      };

      const result = generateDailyWithKeysData(spendDataWithoutTeamId, "Team");
      const entry = result.find((r) => r["Key ID"] === "key1");

      expect(entry?.["Team ID"]).toBe("entity1");
    });

    it("should format spend values correctly", () => {
      const result = generateDailyWithKeysData(mockSpendDataWithKeys, "Team");

      expect(result[0]["Spend ($)"]).toBeDefined();
    });

    it("should handle missing optional token fields", () => {
      const spendDataWithMissingTokens: EntitySpendData = {
        ...mockSpendDataWithKeys,
        results: [
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                entity1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                    prompt_tokens: 600,
                    completion_tokens: 400,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 10.5,
                        api_requests: 100,
                        successful_requests: 95,
                        failed_requests: 5,
                        total_tokens: 1000,
                      },
                      metadata: {
                        team_id: "team-1",
                        key_alias: "alias-1",
                      },
                    },
                  },
                },
              },
            },
          },
        ],
        metadata: mockSpendDataWithKeys.metadata,
      };

      const result = generateDailyWithKeysData(spendDataWithMissingTokens, "Team");
      const key1Entry = result.find((r) => r["Key ID"] === "key1");

      expect(key1Entry?.["Prompt Tokens"]).toBe(0);
      expect(key1Entry?.["Completion Tokens"]).toBe(0);
    });

    it("should handle empty api_key_breakdown", () => {
      const spendDataWithEmptyBreakdown: EntitySpendData = {
        ...mockSpendDataWithKeys,
        results: [
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                entity1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                    prompt_tokens: 600,
                    completion_tokens: 400,
                  },
                  api_key_breakdown: {},
                },
              },
            },
          },
        ],
        metadata: mockSpendDataWithKeys.metadata,
      };

      const result = generateDailyWithKeysData(spendDataWithEmptyBreakdown, "Team");

      expect(result).toHaveLength(0);
    });

    it("should handle multiple keys for same team on same date", () => {
      const result = generateDailyWithKeysData(mockSpendDataWithKeys, "Team");
      const team1Entries = result.filter((r) => r["Team ID"] === "team-1" && r.Date === "2025-01-01");

      expect(team1Entries.length).toBeGreaterThanOrEqual(2);
      const keyIds = team1Entries.map((r) => r["Key ID"]);
      expect(keyIds).toContain("key1");
      expect(keyIds).toContain("key2");
    });
  });

  describe("generateDailyWithModelsData", () => {
    const mockSpendDataWithModels: EntitySpendData = {
      results: [
        {
          date: "2025-01-01",
          breakdown: {
            entities: {
              entity1: {
                metrics: {
                  spend: 10.5,
                  api_requests: 100,
                  successful_requests: 95,
                  failed_requests: 5,
                  total_tokens: 1000,
                  prompt_tokens: 600,
                  completion_tokens: 400,
                  cache_read_input_tokens: 50,
                  cache_creation_input_tokens: 30,
                },
                api_key_breakdown: {
                  key1: {
                    metrics: {
                      spend: 5.0,
                      api_requests: 50,
                      successful_requests: 48,
                      failed_requests: 2,
                      total_tokens: 500,
                    },
                    metadata: {
                      team_id: "team-1",
                    },
                  },
                  key2: {
                    metrics: {
                      spend: 5.5,
                      api_requests: 50,
                      successful_requests: 47,
                      failed_requests: 3,
                      total_tokens: 500,
                    },
                    metadata: {
                      team_id: "team-1",
                    },
                  },
                },
              },
            },
            models: {
              "gpt-4": {
                metrics: {
                  spend: 5.0,
                  api_requests: 50,
                  successful_requests: 48,
                  failed_requests: 2,
                  total_tokens: 500,
                },
              },
              "gpt-3.5-turbo": {
                metrics: {
                  spend: 5.5,
                  api_requests: 50,
                  successful_requests: 47,
                  failed_requests: 3,
                  total_tokens: 500,
                },
              },
            },
          },
        },
      ],
      metadata: {
        total_spend: 10.5,
        total_api_requests: 100,
        total_successful_requests: 95,
        total_failed_requests: 5,
        total_tokens: 1000,
      },
    };

    it("should generate daily breakdown with model data", () => {
      const result = generateDailyWithModelsData(mockSpendDataWithModels, "Team", mockTeamAliasMap);

      expect(result.length).toBeGreaterThan(0);
      expect(result[0]).toHaveProperty("Date");
      expect(result[0]).toHaveProperty("Team");
      expect(result[0]).toHaveProperty("Team ID");
      expect(result[0]).toHaveProperty("Model");
      expect(result[0]).toHaveProperty("Spend ($)");
      expect(result[0]).toHaveProperty("Requests");
      expect(result[0]).toHaveProperty("Successful");
      expect(result[0]).toHaveProperty("Failed");
      expect(result[0]).toHaveProperty("Total Tokens");
    });

    it("should sort data by date ascending", () => {
      const multiDayData: EntitySpendData = {
        results: [
          {
            date: "2025-01-02",
            breakdown: {
              entities: {
                entity1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                    prompt_tokens: 600,
                    completion_tokens: 400,
                    cache_read_input_tokens: 50,
                    cache_creation_input_tokens: 30,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 10.5,
                        api_requests: 100,
                        successful_requests: 95,
                        failed_requests: 5,
                        total_tokens: 1000,
                      },
                      metadata: {
                        team_id: "team-1",
                      },
                    },
                  },
                },
              },
              models: {
                "gpt-4": {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                  },
                },
              },
            },
          },
          ...mockSpendDataWithModels.results,
        ],
        metadata: mockSpendDataWithModels.metadata,
      };

      const result = generateDailyWithModelsData(multiDayData, "Team");

      expect(new Date(result[0].Date).getTime()).toBeLessThanOrEqual(
        new Date(result[result.length - 1].Date).getTime(),
      );
    });

    it("should aggregate model metrics from api key breakdown", () => {
      const result = generateDailyWithModelsData(mockSpendDataWithModels, "Team");

      const gpt4Entry = result.find((r) => r.Model === "gpt-4");
      expect(gpt4Entry).toBeDefined();
      expect(gpt4Entry?.Requests).toBeGreaterThan(0);
    });

    it("should use team alias when available", () => {
      const result = generateDailyWithModelsData(mockSpendDataWithModels, "Team", mockTeamAliasMap);
      const team1Entry = result.find((r) => r["Team ID"] === "team-1");

      expect(team1Entry?.["Team"]).toBe("Team One");
    });

    it("should use dash when team alias is not available", () => {
      const result = generateDailyWithModelsData(mockSpendDataWithModels, "Team");
      const entryWithoutTeamId = result.find((r) => !r["Team ID"] || r["Team ID"] === "-");

      if (entryWithoutTeamId) {
        expect(entryWithoutTeamId["Team"]).toBe("-");
      }
    });

    it("should handle empty models breakdown", () => {
      const spendDataWithoutModels: EntitySpendData = {
        ...mockSpendDataWithModels,
        results: [
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                entity1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                    prompt_tokens: 600,
                    completion_tokens: 400,
                    cache_read_input_tokens: 50,
                    cache_creation_input_tokens: 30,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 10.5,
                        api_requests: 100,
                        successful_requests: 95,
                        failed_requests: 5,
                        total_tokens: 1000,
                      },
                      metadata: {
                        team_id: "team-1",
                      },
                    },
                  },
                },
              },
              models: {},
            },
          },
        ],
        metadata: mockSpendDataWithModels.metadata,
      };

      const result = generateDailyWithModelsData(spendDataWithoutModels, "Team");

      expect(result).toHaveLength(0);
    });
  });

  describe("generateExportData", () => {
    it("should return daily data when scope is daily", () => {
      const result = generateExportData(mockSpendData, "daily", "Team", mockTeamAliasMap);

      expect(result.length).toBeGreaterThan(0);
      expect(result[0]).toHaveProperty("Date");
      expect(result[0]).not.toHaveProperty("Model");
    });

    it("should return daily with keys data when scope is daily_with_keys", () => {
      const mockDataWithKeys: EntitySpendData = {
        ...mockSpendData,
        results: [
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                entity1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                    prompt_tokens: 600,
                    completion_tokens: 400,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 10.5,
                        api_requests: 100,
                        successful_requests: 95,
                        failed_requests: 5,
                        total_tokens: 1000,
                        prompt_tokens: 600,
                        completion_tokens: 400,
                      },
                      metadata: {
                        team_id: "team-1",
                        key_alias: "alias-1",
                      },
                    },
                  },
                },
              },
            },
          },
        ],
        metadata: mockSpendData.metadata,
      };

      const result = generateExportData(mockDataWithKeys, "daily_with_keys", "Team", mockTeamAliasMap);

      expect(result.length).toBeGreaterThan(0);
      expect(result[0]).toHaveProperty("Key Alias");
      expect(result[0]).toHaveProperty("Key ID");
      expect(result[0]).not.toHaveProperty("Model");
    });

    it("should return daily with models data when scope is daily_with_models", () => {
      const mockDataWithModels: EntitySpendData = {
        ...mockSpendData,
        results: [
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                entity1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                    prompt_tokens: 600,
                    completion_tokens: 400,
                    cache_read_input_tokens: 50,
                    cache_creation_input_tokens: 30,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 10.5,
                        api_requests: 100,
                        successful_requests: 95,
                        failed_requests: 5,
                        total_tokens: 1000,
                      },
                      metadata: {
                        team_id: "team-1",
                      },
                    },
                  },
                },
              },
              models: {
                "gpt-4": {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                  },
                },
              },
            },
          },
        ],
        metadata: mockSpendData.metadata,
      };

      const result = generateExportData(mockDataWithModels, "daily_with_models", "Team", mockTeamAliasMap);

      expect(result.length).toBeGreaterThan(0);
      expect(result[0]).toHaveProperty("Model");
    });

    it("should default to daily data for unknown scope", () => {
      const result = generateExportData(mockSpendData, "unknown" as ExportScope, "Team", mockTeamAliasMap);

      expect(result.length).toBeGreaterThan(0);
      expect(result[0]).not.toHaveProperty("Model");
    });
  });

  describe("generateMetadata", () => {
    const mockDateRange: DateRangePickerValue = {
      from: new Date("2025-01-01"),
      to: new Date("2025-01-31"),
    };

    it("should generate metadata with correct structure", () => {
      const result = generateMetadata("team", mockDateRange, [], "daily", mockSpendData);

      expect(result).toHaveProperty("export_date");
      expect(result).toHaveProperty("entity_type");
      expect(result).toHaveProperty("date_range");
      expect(result).toHaveProperty("filters_applied");
      expect(result).toHaveProperty("export_scope");
      expect(result).toHaveProperty("summary");
    });

    it("should include export date as ISO string", () => {
      const result = generateMetadata("team", mockDateRange, [], "daily", mockSpendData);

      expect(result.export_date).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
    });

    it("should include entity type", () => {
      const result = generateMetadata("team", mockDateRange, [], "daily", mockSpendData);

      expect(result.entity_type).toBe("team");
    });

    it("should format date range correctly", () => {
      const result = generateMetadata("team", mockDateRange, [], "daily", mockSpendData);

      expect(result.date_range.from).toBe("2025-01-01T00:00:00.000Z");
      expect(result.date_range.to).toBe("2025-01-31T00:00:00.000Z");
    });

    it("should handle missing date range values", () => {
      const incompleteDateRange: DateRangePickerValue = {
        from: undefined,
        to: undefined,
      };

      const result = generateMetadata("team", incompleteDateRange, [], "daily", mockSpendData);

      expect(result.date_range.from).toBeUndefined();
      expect(result.date_range.to).toBeUndefined();
    });

    it("should set filters_applied to None when empty", () => {
      const result = generateMetadata("team", mockDateRange, [], "daily", mockSpendData);

      expect(result.filters_applied).toBe("None");
    });

    it("should include filters when provided", () => {
      const result = generateMetadata("team", mockDateRange, ["filter1", "filter2"], "daily", mockSpendData);

      expect(result.filters_applied).toEqual(["filter1", "filter2"]);
    });

    it("should include export scope", () => {
      const result = generateMetadata("team", mockDateRange, [], "daily_with_models", mockSpendData);

      expect(result.export_scope).toBe("daily_with_models");
    });

    it("should include summary metrics from spend data", () => {
      const result = generateMetadata("team", mockDateRange, [], "daily", mockSpendData);

      expect(result.summary.total_spend).toBe(46.0);
      expect(result.summary.total_requests).toBe(450);
      expect(result.summary.successful_requests).toBe(430);
      expect(result.summary.failed_requests).toBe(20);
      expect(result.summary.total_tokens).toBe(4500);
    });
  });

  describe("handleExportCSV", () => {
    beforeEach(() => {
      document.body.innerHTML = "";
      window.URL.createObjectURL = vi.fn(() => "blob:mock-url");
      window.URL.revokeObjectURL = vi.fn();
    });

    afterEach(() => {
      vi.restoreAllMocks();
    });

    it("should create CSV file and trigger download", () => {
      const createObjectURLSpy = vi.spyOn(window.URL, "createObjectURL").mockReturnValue("blob:mock-url");
      const revokeObjectURLSpy = vi.spyOn(window.URL, "revokeObjectURL");
      const createElementSpy = vi.spyOn(document, "createElement");
      const appendChildSpy = vi.spyOn(document.body, "appendChild");
      const removeChildSpy = vi.spyOn(document.body, "removeChild");

      handleExportCSV(mockSpendData, "daily", "Team", "team", mockTeamAliasMap);

      expect(Papa.unparse).toHaveBeenCalled();
      expect(createObjectURLSpy).toHaveBeenCalled();
      expect(createElementSpy).toHaveBeenCalledWith("a");
      expect(appendChildSpy).toHaveBeenCalled();
      expect(removeChildSpy).toHaveBeenCalled();
    });

    it("should generate correct filename", () => {
      const anchorElement = document.createElement("a");
      const createElementSpy = vi.spyOn(document, "createElement").mockReturnValue(anchorElement);

      const today = new Date().toISOString().split("T")[0];

      handleExportCSV(mockSpendData, "daily", "Team", "team", mockTeamAliasMap);

      expect(anchorElement.download).toBe(`team_usage_daily_${today}.csv`);
    });

    it("should create blob with correct type", () => {
      let blobType = "";
      const originalBlob = window.Blob;

      window.Blob = class extends Blob {
        constructor(parts?: BlobPart[] | undefined, options?: BlobPropertyBag | undefined) {
          super(parts, options);
          if (options?.type) {
            blobType = options.type;
          }
        }
      } as any;

      handleExportCSV(mockSpendData, "daily", "Team", "team", mockTeamAliasMap);

      expect(blobType).toBe("text/csv;charset=utf-8;");

      window.Blob = originalBlob;
    });
  });

  describe("handleExportJSON", () => {
    beforeEach(() => {
      document.body.innerHTML = "";
      window.URL.createObjectURL = vi.fn(() => "blob:mock-url");
      window.URL.revokeObjectURL = vi.fn();
    });

    afterEach(() => {
      vi.restoreAllMocks();
    });

    it("should create JSON file and trigger download", () => {
      const createObjectURLSpy = vi.spyOn(window.URL, "createObjectURL").mockReturnValue("blob:mock-url");
      const revokeObjectURLSpy = vi.spyOn(window.URL, "revokeObjectURL");
      const createElementSpy = vi.spyOn(document, "createElement");
      const appendChildSpy = vi.spyOn(document.body, "appendChild");
      const removeChildSpy = vi.spyOn(document.body, "removeChild");

      const mockDateRange: DateRangePickerValue = {
        from: new Date("2025-01-01"),
        to: new Date("2025-01-31"),
      };

      handleExportJSON(mockSpendData, "daily", "Team", "team", mockDateRange, [], mockTeamAliasMap);

      expect(createObjectURLSpy).toHaveBeenCalled();
      expect(createElementSpy).toHaveBeenCalledWith("a");
      expect(appendChildSpy).toHaveBeenCalled();
      expect(removeChildSpy).toHaveBeenCalled();
    });

    it("should generate correct filename", () => {
      const anchorElement = document.createElement("a");
      const createElementSpy = vi.spyOn(document, "createElement").mockReturnValue(anchorElement);

      const today = new Date().toISOString().split("T")[0];
      const mockDateRange: DateRangePickerValue = {
        from: new Date("2025-01-01"),
        to: new Date("2025-01-31"),
      };

      handleExportJSON(mockSpendData, "daily", "Team", "team", mockDateRange, [], mockTeamAliasMap);

      expect(anchorElement.download).toBe(`team_usage_daily_${today}.json`);
    });

    it("should create blob with correct type", () => {
      let blobType = "";
      const originalBlob = window.Blob;

      window.Blob = class extends Blob {
        constructor(parts?: BlobPart[] | undefined, options?: BlobPropertyBag | undefined) {
          super(parts, options);
          if (options?.type) {
            blobType = options.type;
          }
        }
      } as any;

      const mockDateRange: DateRangePickerValue = {
        from: new Date("2025-01-01"),
        to: new Date("2025-01-31"),
      };

      handleExportJSON(mockSpendData, "daily", "Team", "team", mockDateRange, [], mockTeamAliasMap);

      expect(blobType).toBe("application/json");

      window.Blob = originalBlob;
    });

    it("should include metadata and data in JSON export", () => {
      let jsonString = "";
      const originalBlob = window.Blob;

      window.Blob = class extends Blob {
        constructor(parts?: BlobPart[] | undefined, options?: BlobPropertyBag | undefined) {
          super(parts, options);
          if (parts && parts[0]) {
            jsonString = parts[0] as string;
          }
        }
      } as any;

      const mockDateRange: DateRangePickerValue = {
        from: new Date("2025-01-01"),
        to: new Date("2025-01-31"),
      };

      handleExportJSON(mockSpendData, "daily", "Team", "team", mockDateRange, ["filter1"], mockTeamAliasMap);

      const exportObject = JSON.parse(jsonString);
      expect(exportObject).toHaveProperty("metadata");
      expect(exportObject).toHaveProperty("data");
      expect(exportObject.metadata.entity_type).toBe("team");
      expect(exportObject.metadata.filters_applied).toEqual(["filter1"]);

      window.Blob = originalBlob;
    });
  });
});
