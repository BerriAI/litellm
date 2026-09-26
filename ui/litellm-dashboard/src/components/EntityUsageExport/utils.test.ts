// @vitest-environment jsdom

import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import Papa from "papaparse";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { EntitySpendData, ExportScope } from "./types";
import {
  generateDailyData,
  generateDailyWithKeysData,
  generateDailyWithModelsData,
  generateDailyWithUsersData,
  generateExportData,
  generateMetadata,
  getEntityBreakdown,
  handleExportCSV,
  handleExportJSON,
  handleServerExport,
  resolveEntities,
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
  // Entity keys match team_ids because that's how the backend shapes team exports
  // (breakdown.entities is keyed by team_id). The fix under test uses the entity key
  // directly for display, so the key_alias/team_id in api_key_breakdown metadata is
  // no longer consulted — it's retained here only to mirror real payload shape.
  const mockSpendData: EntitySpendData = {
    results: [
      {
        date: "2025-01-01",
        breakdown: {
          entities: {
            "team-1": {
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
            "team-2": {
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
            "team-1": {
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

  const usersFixture: EntitySpendData = {
    results: [
      {
        date: "2025-03-01",
        breakdown: {
          entities: {
            "team-1": {
              metrics: {
                spend: 16.5,
                api_requests: 165,
                successful_requests: 156,
                failed_requests: 9,
                total_tokens: 1650,
                prompt_tokens: 940,
                completion_tokens: 710,
                cache_read_input_tokens: 90,
                cache_creation_input_tokens: 60,
              },
              api_key_breakdown: {
                kA: {
                  metrics: {
                    spend: 1.1,
                    api_requests: 11,
                    successful_requests: 10,
                    failed_requests: 1,
                    total_tokens: 110,
                    prompt_tokens: 60,
                    completion_tokens: 50,
                    cache_read_input_tokens: 6,
                    cache_creation_input_tokens: 4,
                  },
                  metadata: {
                    team_id: "team-1",
                    key_alias: "alice-key",
                    user_id: "u1",
                    user_email: "a@x",
                  },
                },
                kB: {
                  metrics: {
                    spend: 2.2,
                    api_requests: 22,
                    successful_requests: 20,
                    failed_requests: 2,
                    total_tokens: 220,
                    prompt_tokens: 130,
                    completion_tokens: 90,
                    cache_read_input_tokens: 12,
                    cache_creation_input_tokens: 8,
                  },
                  metadata: {
                    team_id: "team-1",
                    user_id: "u1",
                    user_email: "a@x",
                  },
                },
                kC: {
                  metrics: {
                    spend: 3.3,
                    api_requests: 33,
                    successful_requests: 31,
                    failed_requests: 2,
                    total_tokens: 330,
                    prompt_tokens: 190,
                    completion_tokens: 140,
                    cache_read_input_tokens: 18,
                    cache_creation_input_tokens: 12,
                  },
                  metadata: {
                    team_id: "team-1",
                    user_id: "u2",
                    user_email: null,
                  },
                },
                kD: {
                  metrics: {
                    spend: 4.4,
                    api_requests: 44,
                    successful_requests: 42,
                    failed_requests: 2,
                    total_tokens: 440,
                    prompt_tokens: 250,
                    completion_tokens: 190,
                    cache_read_input_tokens: 24,
                    cache_creation_input_tokens: 16,
                  },
                  metadata: {
                    team_id: "team-1",
                    user_id: null,
                  },
                },
                kE: {
                  metrics: {
                    spend: 5.5,
                    api_requests: 55,
                    successful_requests: 53,
                    failed_requests: 2,
                    total_tokens: 550,
                    prompt_tokens: 310,
                    completion_tokens: 240,
                    cache_read_input_tokens: 30,
                    cache_creation_input_tokens: 20,
                  },
                  metadata: {
                    team_id: "team-1",
                    user_id: "u3",
                    key_exists: false,
                  },
                },
              },
            },
            "team-2": {
              metrics: {
                spend: 6.6,
                api_requests: 66,
                successful_requests: 64,
                failed_requests: 2,
                total_tokens: 660,
                prompt_tokens: 370,
                completion_tokens: 290,
                cache_read_input_tokens: 36,
                cache_creation_input_tokens: 24,
              },
              api_key_breakdown: {
                kF: {
                  metrics: {
                    spend: 6.6,
                    api_requests: 66,
                    successful_requests: 64,
                    failed_requests: 2,
                    total_tokens: 660,
                    prompt_tokens: 370,
                    completion_tokens: 290,
                    cache_read_input_tokens: 36,
                    cache_creation_input_tokens: 24,
                  },
                  metadata: {
                    team_id: "team-2",
                    user_id: "u1",
                    user_email: "a@x",
                  },
                },
              },
            },
          },
        },
      },
      {
        date: "2025-03-02",
        breakdown: {
          entities: {
            "team-1": {
              metrics: {
                spend: 7.7,
                api_requests: 77,
                successful_requests: 75,
                failed_requests: 2,
                total_tokens: 770,
                prompt_tokens: 430,
                completion_tokens: 340,
                cache_read_input_tokens: 42,
                cache_creation_input_tokens: 28,
              },
              api_key_breakdown: {
                kA: {
                  metrics: {
                    spend: 7.7,
                    api_requests: 77,
                    successful_requests: 75,
                    failed_requests: 2,
                    total_tokens: 770,
                    prompt_tokens: 430,
                    completion_tokens: 340,
                    cache_read_input_tokens: 42,
                    cache_creation_input_tokens: 28,
                  },
                  metadata: {
                    team_id: "team-1",
                    key_alias: "alice-key",
                    user_id: "u1",
                    user_email: "a@x",
                  },
                },
              },
            },
          },
        },
      },
    ],
    metadata: {
      total_spend: 30.8,
      total_api_requests: 308,
      total_successful_requests: 295,
      total_failed_requests: 13,
      total_tokens: 3080,
    },
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

    it("should use entity key as alias when no team alias map is provided", () => {
      // Non-team exports (tags, orgs, customers, …) pass no teamAliasMap.
      // For teams, this is also the fallback when a team is missing from the map.
      const result = getEntityBreakdown(mockSpendData);
      const entity1 = result.find((e) => e.metadata.id === "team-1");

      expect(entity1?.metadata.alias).toBe("team-1");
    });

    it("should use team alias map to resolve alias from entity key", () => {
      const spendDataWithoutAlias: EntitySpendData = {
        ...mockSpendData,
        results: [
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                "team-1": {
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
                "team-1": {
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
      expect(result[0]).toHaveProperty("Cache Read Input Tokens");
      expect(result[0]).toHaveProperty("Cache Creation Input Tokens");
    });

    it("should export exact cache token values per entity per day", () => {
      const result = generateDailyData(mockSpendData, "Team", mockTeamAliasMap);

      const day1Team1 = result.find((r) => r.Date === "2025-01-01" && r["Team ID"] === "team-1");
      const day1Team2 = result.find((r) => r.Date === "2025-01-01" && r["Team ID"] === "team-2");
      const day2Team1 = result.find((r) => r.Date === "2025-01-02" && r["Team ID"] === "team-1");

      expect(day1Team1?.["Cache Read Input Tokens"]).toBe(50);
      expect(day1Team1?.["Cache Creation Input Tokens"]).toBe(30);
      expect(day1Team2?.["Cache Read Input Tokens"]).toBe(100);
      expect(day1Team2?.["Cache Creation Input Tokens"]).toBe(60);
      expect(day2Team1?.["Cache Read Input Tokens"]).toBe(75);
      expect(day2Team1?.["Cache Creation Input Tokens"]).toBe(45);
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

    it("should fall back to the entity key when there is no team alias mapping", () => {
      // e.g. tag/org/customer exports where teamAliasMap has no entry for the entity,
      // or a team that isn't in the alias map — the entity key itself is the label.
      const spendDataWithoutAlias: EntitySpendData = {
        ...mockSpendData,
        results: [
          {
            date: "2025-01-01",
            breakdown: {
              entities: {
                "my-tag": {
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

      const result = generateDailyData(spendDataWithoutAlias, "Tag");
      const entry = result[0];

      expect(entry["Tag ID"]).toBe("my-tag");
      expect(entry["Tag"]).toBe("my-tag");
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
      expect(result[0]["Cache Read Input Tokens"]).toBe(0);
      expect(result[0]["Cache Creation Input Tokens"]).toBe(0);
    });
  });

  describe("generateDailyWithKeysData", () => {
    const mockSpendDataWithKeys: EntitySpendData = {
      results: [
        {
          date: "2025-01-01",
          breakdown: {
            entities: {
              "team-1": {
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
              "team-2": {
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
              "team-1": {
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
      expect(result[0]).toHaveProperty("Cache Read Input Tokens");
      expect(result[0]).toHaveProperty("Cache Creation Input Tokens");
    });

    it("should export and aggregate cache token values per key", () => {
      const makeDay = (cacheRead: number, cacheCreation: number) => ({
        date: "2025-01-01",
        breakdown: {
          entities: {
            "team-1": {
              metrics: {
                spend: 5.0,
                api_requests: 50,
                successful_requests: 50,
                failed_requests: 0,
                total_tokens: 500,
                prompt_tokens: 300,
                completion_tokens: 200,
                cache_read_input_tokens: cacheRead,
                cache_creation_input_tokens: cacheCreation,
              },
              api_key_breakdown: {
                key1: {
                  metrics: {
                    spend: 5.0,
                    api_requests: 50,
                    successful_requests: 50,
                    failed_requests: 0,
                    total_tokens: 500,
                    prompt_tokens: 300,
                    completion_tokens: 200,
                    cache_read_input_tokens: cacheRead,
                    cache_creation_input_tokens: cacheCreation,
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
      });

      const spendDataWithCache: EntitySpendData = {
        results: [makeDay(40, 25), makeDay(10, 5)],
        metadata: mockSpendDataWithKeys.metadata,
      };

      const result = generateDailyWithKeysData(spendDataWithCache, "Team");

      expect(result).toHaveLength(1);
      expect(result[0]["Cache Read Input Tokens"]).toBe(50);
      expect(result[0]["Cache Creation Input Tokens"]).toBe(30);
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
      expect(key1Entry?.["Cache Read Input Tokens"]).toBe(0);
      expect(key1Entry?.["Cache Creation Input Tokens"]).toBe(0);
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

    it("should emit key owner columns right after Key ID", () => {
      const result = generateDailyWithKeysData(usersFixture, "Team");

      const columnNames = Object.keys(result[0]);
      expect(columnNames[columnNames.indexOf("Key ID") + 1]).toBe("User ID");
      expect(columnNames[columnNames.indexOf("User ID") + 1]).toBe("User Email");

      const kARow = result.find((r) => r["Key ID"] === "kA" && r.Date === "2025-03-01");
      expect(kARow?.["User ID"]).toBe("u1");
      expect(kARow?.["User Email"]).toBe("a@x");
      expect(kARow?.["Key Alias"]).toBe("alice-key");

      const kCRow = result.find((r) => r["Key ID"] === "kC");
      expect(kCRow?.["User ID"]).toBe("u2");
      expect(kCRow?.["User Email"]).toBe("-");

      const kDRow = result.find((r) => r["Key ID"] === "kD");
      expect(kDRow?.["User ID"]).toBe("-");
      expect(kDRow?.["User Email"]).toBe("-");
    });
  });

  describe("generateDailyWithModelsData", () => {
    const mockSpendDataWithModels: EntitySpendData = {
      results: [
        {
          date: "2025-01-01",
          breakdown: {
            entities: {
              "team-1": {
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
                api_key_breakdown: {
                  key1: {
                    metrics: {
                      spend: 5.0,
                      api_requests: 50,
                      successful_requests: 48,
                      failed_requests: 2,
                      total_tokens: 500,
                    },
                    metadata: { team_id: "team-1" },
                  },
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
                api_key_breakdown: {
                  key2: {
                    metrics: {
                      spend: 5.5,
                      api_requests: 50,
                      successful_requests: 47,
                      failed_requests: 3,
                      total_tokens: 500,
                    },
                    metadata: { team_id: "team-1" },
                  },
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
      expect(result[0]).toHaveProperty("Prompt Tokens");
      expect(result[0]).toHaveProperty("Completion Tokens");
      expect(result[0]).toHaveProperty("Cache Read Input Tokens");
      expect(result[0]).toHaveProperty("Cache Creation Input Tokens");
    });

    it("should export prompt, completion, and cache token values summed across keys for the same model", () => {
      const data: EntitySpendData = {
        results: [
          {
            date: "2025-03-01",
            breakdown: {
              entities: {
                "team-1": {
                  metrics: {
                    spend: 5.0,
                    api_requests: 25,
                    successful_requests: 25,
                    failed_requests: 0,
                    total_tokens: 1050,
                    prompt_tokens: 750,
                    completion_tokens: 300,
                    cache_read_input_tokens: 450,
                    cache_creation_input_tokens: 200,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 2.0,
                        api_requests: 10,
                        successful_requests: 10,
                        failed_requests: 0,
                        total_tokens: 700,
                      },
                      metadata: { team_id: "team-1" },
                    },
                    key2: {
                      metrics: {
                        spend: 3.0,
                        api_requests: 15,
                        successful_requests: 15,
                        failed_requests: 0,
                        total_tokens: 350,
                      },
                      metadata: { team_id: "team-1" },
                    },
                  },
                },
              },
              models: {
                "claude-sonnet-4-5": {
                  metrics: {
                    spend: 5.0,
                    api_requests: 25,
                    successful_requests: 25,
                    failed_requests: 0,
                    total_tokens: 1050,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 2.0,
                        api_requests: 10,
                        successful_requests: 10,
                        failed_requests: 0,
                        total_tokens: 700,
                        prompt_tokens: 500,
                        completion_tokens: 200,
                        cache_read_input_tokens: 300,
                        cache_creation_input_tokens: 120,
                      },
                      metadata: { team_id: "team-1" },
                    },
                    key2: {
                      metrics: {
                        spend: 3.0,
                        api_requests: 15,
                        successful_requests: 15,
                        failed_requests: 0,
                        total_tokens: 350,
                        prompt_tokens: 250,
                        completion_tokens: 100,
                        cache_read_input_tokens: 150,
                        cache_creation_input_tokens: 80,
                      },
                      metadata: { team_id: "team-1" },
                    },
                  },
                },
              },
            },
          },
        ],
        metadata: {
          total_spend: 5.0,
          total_api_requests: 25,
          total_successful_requests: 25,
          total_failed_requests: 0,
          total_tokens: 1050,
        },
      };

      const result = generateDailyWithModelsData(data, "Team");

      expect(result).toHaveLength(1);
      expect(result[0].Model).toBe("claude-sonnet-4-5");
      expect(result[0]["Total Tokens"]).toBe(1050);
      expect(result[0]["Prompt Tokens"]).toBe(750);
      expect(result[0]["Completion Tokens"]).toBe(300);
      expect(result[0]["Cache Read Input Tokens"]).toBe(450);
      expect(result[0]["Cache Creation Input Tokens"]).toBe(200);
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
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 10.5,
                        api_requests: 100,
                        successful_requests: 95,
                        failed_requests: 5,
                        total_tokens: 1000,
                      },
                      metadata: { team_id: "team-1" },
                    },
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

    it("should attribute each model only its own per-key spend", () => {
      const result = generateDailyWithModelsData(mockSpendDataWithModels, "Team");

      const gpt4Entry = result.find((r) => r.Model === "gpt-4");
      const gpt35Entry = result.find((r) => r.Model === "gpt-3.5-turbo");

      expect(gpt4Entry?.["Spend ($)"]).toBe("5.0000");
      expect(gpt4Entry?.Requests).toBe(50);
      expect(gpt4Entry?.["Total Tokens"]).toBe(500);

      expect(gpt35Entry?.["Spend ($)"]).toBe("5.5000");
      expect(gpt35Entry?.Requests).toBe(50);
      expect(gpt35Entry?.["Total Tokens"]).toBe(500);
    });

    it("should not duplicate a user's spend across every model (regression for LIT overcount)", () => {
      // One user, one key, that key used two models. The entity-level api_key_breakdown
      // carries the key's total (8.0) across both models; each model's api_key_breakdown
      // carries only that model's share (3.0 + 5.0). The per-model rows must sum back to
      // the user-day total, not repeat the total once per model.
      const data: EntitySpendData = {
        results: [
          {
            date: "2025-02-14",
            breakdown: {
              entities: {
                user1: {
                  metrics: {
                    spend: 8.0,
                    api_requests: 80,
                    successful_requests: 78,
                    failed_requests: 2,
                    total_tokens: 800,
                    prompt_tokens: 500,
                    completion_tokens: 300,
                    cache_read_input_tokens: 0,
                    cache_creation_input_tokens: 0,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 8.0,
                        api_requests: 80,
                        successful_requests: 78,
                        failed_requests: 2,
                        total_tokens: 800,
                      },
                      metadata: { team_id: "team-1" },
                    },
                  },
                },
              },
              models: {
                "claude-3-haiku": {
                  metrics: {
                    spend: 3.0,
                    api_requests: 30,
                    successful_requests: 29,
                    failed_requests: 1,
                    total_tokens: 300,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 3.0,
                        api_requests: 30,
                        successful_requests: 29,
                        failed_requests: 1,
                        total_tokens: 300,
                      },
                      metadata: { team_id: "team-1" },
                    },
                  },
                },
                "claude-sonnet-4-5": {
                  metrics: {
                    spend: 5.0,
                    api_requests: 50,
                    successful_requests: 49,
                    failed_requests: 1,
                    total_tokens: 500,
                  },
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 5.0,
                        api_requests: 50,
                        successful_requests: 49,
                        failed_requests: 1,
                        total_tokens: 500,
                      },
                      metadata: { team_id: "team-1" },
                    },
                  },
                },
              },
            },
          },
        ],
        metadata: {
          total_spend: 8.0,
          total_api_requests: 80,
          total_successful_requests: 78,
          total_failed_requests: 2,
          total_tokens: 800,
        },
      };

      const result = generateDailyWithModelsData(data, "User");

      expect(result).toHaveLength(2);

      const haiku = result.find((r) => r.Model === "claude-3-haiku");
      const sonnet = result.find((r) => r.Model === "claude-sonnet-4-5");

      expect(haiku?.["Spend ($)"]).toBe("3.0000");
      expect(sonnet?.["Spend ($)"]).toBe("5.0000");

      const totalSpend = result.reduce((sum, r) => sum + parseFloat(r["Spend ($)"].replace(/,/g, "")), 0);
      const totalRequests = result.reduce((sum, r) => sum + r.Requests, 0);
      const totalTokens = result.reduce((sum, r) => sum + r["Total Tokens"], 0);

      expect(totalSpend).toBeCloseTo(8.0, 4);
      expect(totalRequests).toBe(80);
      expect(totalTokens).toBe(800);
    });

    it("should omit models the user never called instead of fanning out", () => {
      // A second key (key2) belongs to a different user and is the only caller of
      // gpt-3.5-turbo. user1 only used key1 -> gpt-4. user1 must get exactly one row.
      const data: EntitySpendData = {
        results: [
          {
            date: "2025-02-14",
            breakdown: {
              entities: {
                user1: {
                  metrics: {
                    spend: 5.0,
                    api_requests: 50,
                    successful_requests: 48,
                    failed_requests: 2,
                    total_tokens: 500,
                    prompt_tokens: 300,
                    completion_tokens: 200,
                    cache_read_input_tokens: 0,
                    cache_creation_input_tokens: 0,
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
                      metadata: { team_id: "team-1" },
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
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 5.0,
                        api_requests: 50,
                        successful_requests: 48,
                        failed_requests: 2,
                        total_tokens: 500,
                      },
                      metadata: { team_id: "team-1" },
                    },
                  },
                },
                "gpt-3.5-turbo": {
                  metrics: {
                    spend: 9.0,
                    api_requests: 90,
                    successful_requests: 90,
                    failed_requests: 0,
                    total_tokens: 900,
                  },
                  api_key_breakdown: {
                    key2: {
                      metrics: {
                        spend: 9.0,
                        api_requests: 90,
                        successful_requests: 90,
                        failed_requests: 0,
                        total_tokens: 900,
                      },
                      metadata: { team_id: "team-2" },
                    },
                  },
                },
              },
            },
          },
        ],
        metadata: {
          total_spend: 14.0,
          total_api_requests: 140,
          total_successful_requests: 138,
          total_failed_requests: 2,
          total_tokens: 1400,
        },
      };

      const result = generateDailyWithModelsData(data, "User");

      expect(result).toHaveLength(1);
      expect(result[0].Model).toBe("gpt-4");
      expect(result[0]["Spend ($)"]).toBe("5.0000");
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
                  api_key_breakdown: {
                    key1: {
                      metrics: {
                        spend: 10.5,
                        api_requests: 100,
                        successful_requests: 95,
                        failed_requests: 5,
                        total_tokens: 1000,
                      },
                      metadata: { team_id: "team-1" },
                    },
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

    it("should include total_flat_cost and total_cost in summary when total_flat_cost is present", () => {
      const spendWithFlat: EntitySpendData = {
        ...mockSpendData,
        metadata: { ...mockSpendData.metadata, total_flat_cost: 6.45 },
      };
      const result = generateMetadata("team", mockDateRange, [], "daily", spendWithFlat);
      expect(result.summary.total_flat_cost).toBeCloseTo(6.45, 4);
      expect(result.summary.total_cost).toBeCloseTo(46.0 + 6.45, 4);
    });

    it("should omit total_flat_cost and total_cost when total_flat_cost is zero", () => {
      const zeroFlat = { ...mockSpendData, metadata: { ...mockSpendData.metadata, total_flat_cost: 0 } };
      const result = generateMetadata("team", mockDateRange, [], "daily", zeroFlat);
      expect(result.summary.total_flat_cost).toBeUndefined();
      expect(result.summary.total_cost).toBeUndefined();
    });
  });

  describe("generateDailyData PTU flat cost", () => {
    const dayWithFlat: EntitySpendData = {
      results: [
        {
          date: "2025-01-01",
          breakdown: {
            entities: {
              "team-1": {
                metrics: {
                  spend: 10,
                  flat_cost: 6.45,
                  api_requests: 50,
                  successful_requests: 50,
                  failed_requests: 0,
                  total_tokens: 500,
                  prompt_tokens: 300,
                  completion_tokens: 200,
                  cache_read_input_tokens: 0,
                  cache_creation_input_tokens: 0,
                },
                api_key_breakdown: {},
              },
            },
          },
        },
      ],
      metadata: {
        total_spend: 10,
        total_flat_cost: 6.45,
        total_api_requests: 50,
        total_successful_requests: 50,
        total_failed_requests: 0,
        total_tokens: 500,
      },
    };

    it("includes Flat Cost ($) and Total Cost ($) columns when total_flat_cost is present", () => {
      const rows = generateDailyData(dayWithFlat, "Team", {});
      expect(rows).toHaveLength(1);
      expect(rows[0]).toHaveProperty("Flat Cost ($)");
      expect(rows[0]).toHaveProperty("Total Cost ($)");
      expect(rows[0]["Flat Cost ($)"]).toBe("6.4500");
      expect(rows[0]["Total Cost ($)"]).toBe("16.4500");
    });

    it("does not include Flat Cost / Total Cost columns when total_flat_cost is zero", () => {
      const spendWithoutFlat: EntitySpendData = {
        ...dayWithFlat,
        metadata: {
          total_spend: 10,
          total_api_requests: 50,
          total_successful_requests: 50,
          total_failed_requests: 0,
          total_tokens: 500,
          total_flat_cost: 0,
        },
      };
      const rows = generateDailyData(spendWithoutFlat, "User", {});
      expect(rows).toHaveLength(1);
      expect(rows[0]).not.toHaveProperty("Flat Cost ($)");
      expect(rows[0]).not.toHaveProperty("Total Cost ($)");
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
      vi.spyOn(window.URL, "revokeObjectURL");
      const createElementSpy = vi.spyOn(document, "createElement");
      const appendChildSpy = vi.spyOn(document.body, "appendChild");
      const removeChildSpy = vi.spyOn(document.body, "removeChild");

      handleExportCSV(mockSpendData, "daily", "Team", "team", mockTeamAliasMap);

      const unparsedRows = vi.mocked(Papa.unparse).mock.calls[0][0] as Record<string, unknown>[];
      expect(unparsedRows).toHaveLength(3);
      const day1Team1 = unparsedRows.find((r) => r["Date"] === "2025-01-01" && r["Team ID"] === "team-1");
      expect(day1Team1?.["Cache Read Input Tokens"]).toBe(50);

      const exportedBlob = createObjectURLSpy.mock.calls[0][0] as Blob;
      expect(exportedBlob.type).toBe("text/csv;charset=utf-8;");

      expect(createElementSpy).toHaveBeenCalledWith("a");
      const attached = appendChildSpy.mock.calls[0][0] as HTMLAnchorElement;
      expect(attached.download).toMatch(/^team_usage_daily_.*\.csv$/);
      expect(removeChildSpy).toHaveBeenCalledWith(attached);
    });

    it("should generate correct filename", () => {
      const anchorElement = document.createElement("a");
      vi.spyOn(document, "createElement").mockReturnValue(anchorElement);

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

    it("should generate the daily_with_users filename and include User ID in the rows", () => {
      const anchorElement = document.createElement("a");
      vi.spyOn(document, "createElement").mockReturnValue(anchorElement);
      vi.useFakeTimers();
      vi.setSystemTime(new Date("2025-03-01T12:00:00Z"));

      handleExportCSV(usersFixture, "daily_with_users", "Team", "team", mockTeamAliasMap);
      vi.useRealTimers();

      expect(anchorElement.download).toBe("team_usage_daily_with_users_2025-03-01.csv");

      const unparsedRows = vi.mocked(Papa.unparse).mock.calls[0][0] as Record<string, unknown>[];
      expect(unparsedRows[0]).toHaveProperty("User ID");
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
      vi.spyOn(window.URL, "revokeObjectURL");
      const createElementSpy = vi.spyOn(document, "createElement");
      const appendChildSpy = vi.spyOn(document.body, "appendChild");
      const removeChildSpy = vi.spyOn(document.body, "removeChild");

      const mockDateRange: DateRangePickerValue = {
        from: new Date("2025-01-01"),
        to: new Date("2025-01-31"),
      };

      handleExportJSON(mockSpendData, "daily", "Team", "team", mockDateRange, [], mockTeamAliasMap);

      const exportedBlob = createObjectURLSpy.mock.calls[0][0] as Blob;
      expect(exportedBlob.type).toBe("application/json");

      expect(createElementSpy).toHaveBeenCalledWith("a");
      const attached = appendChildSpy.mock.calls[0][0] as HTMLAnchorElement;
      expect(attached.download).toMatch(/^team_usage_daily_.*\.json$/);
      expect(removeChildSpy).toHaveBeenCalledWith(attached);
    });

    it("should generate correct filename", () => {
      const anchorElement = document.createElement("a");
      vi.spyOn(document, "createElement").mockReturnValue(anchorElement);

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

  describe("resolveEntities and aggregated endpoint fallback", () => {
    // Simulates the response from /user/daily/activity/aggregated which has
    // empty entities but populated api_keys at the breakdown level.
    // Derived from mockSpendData: flatten all entities' api_key_breakdowns
    // into top-level api_keys, clear entities, and add a second key for team-1
    // to test multi-key grouping.
    const aggregatedSpendData: EntitySpendData = {
      ...mockSpendData,
      results: mockSpendData.results.slice(0, 1).map((day) => ({
        ...day,
        breakdown: {
          entities: {},
          api_keys: {
            ...Object.fromEntries(
              Object.values(day.breakdown.entities as Record<string, any>).flatMap((e: any) =>
                Object.entries(e.api_key_breakdown || {}),
              ),
            ),
            // Extra key on team-1 to test multi-key-per-team aggregation
            key1b: {
              metrics: { spend: 5, api_requests: 50, successful_requests: 48, failed_requests: 2, total_tokens: 500 },
              metadata: { team_id: "team-1", key_alias: "staging-key" },
            },
          },
          models: {
            "gpt-4": {
              metrics: { spend: 35.8, api_requests: 350, total_tokens: 3500 },
              api_key_breakdown: {
                key1: {
                  metrics: {
                    spend: 10.5,
                    api_requests: 100,
                    successful_requests: 95,
                    failed_requests: 5,
                    total_tokens: 1000,
                  },
                  metadata: { team_id: "team-1" },
                },
                key1b: {
                  metrics: {
                    spend: 5,
                    api_requests: 50,
                    successful_requests: 48,
                    failed_requests: 2,
                    total_tokens: 500,
                  },
                  metadata: { team_id: "team-1" },
                },
                key2: {
                  metrics: {
                    spend: 20.3,
                    api_requests: 200,
                    successful_requests: 195,
                    failed_requests: 5,
                    total_tokens: 2000,
                  },
                  metadata: { team_id: "team-2" },
                },
              },
            },
          },
        },
      })),
    };

    describe("resolveEntities", () => {
      it("should return entities when populated", () => {
        const breakdown = {
          entities: { e1: { metrics: { spend: 1 } } },
          api_keys: { k1: { metrics: { spend: 2 }, metadata: { team_id: "t1" } } },
        };
        const result = resolveEntities(breakdown);
        expect(result).toBe(breakdown.entities);
      });

      it("should aggregate api_keys into entities when entities is empty", () => {
        const breakdown = aggregatedSpendData.results[0].breakdown;
        const result = resolveEntities(breakdown);

        // Two teams: team-1 (key1+key2) and team-2 (key3)
        expect(Object.keys(result)).toHaveLength(2);
        expect(result["team-1"]).toBeDefined();
        expect(result["team-2"]).toBeDefined();

        // team-1 spend = 10.5 (key1) + 5 (key1b)
        expect(result["team-1"].metrics.spend).toBe(15.5);
        expect(result["team-1"].metrics.api_requests).toBe(150);
        expect(result["team-1"].metrics.total_tokens).toBe(1500);

        // team-2 spend = 20.3 (key2)
        expect(result["team-2"].metrics.spend).toBe(20.3);
        expect(result["team-2"].metrics.api_requests).toBe(200);
      });

      it("should use 'Unassigned' for keys without team_id", () => {
        const breakdown = {
          entities: {},
          api_keys: {
            k1: {
              metrics: { spend: 7, api_requests: 10, successful_requests: 10, failed_requests: 0, total_tokens: 100 },
              metadata: {},
            },
          },
        };
        const result = resolveEntities(breakdown);
        expect(result["Unassigned"]).toBeDefined();
        expect(result["Unassigned"].metrics.spend).toBe(7);
      });

      it("should handle missing or empty api_keys gracefully", () => {
        expect(Object.keys(resolveEntities({ entities: {}, api_keys: {} }))).toHaveLength(0);
        expect(Object.keys(resolveEntities({ entities: {} }))).toHaveLength(0);
      });

      it("should preserve api_key_breakdown on aggregated entities", () => {
        const breakdown = aggregatedSpendData.results[0].breakdown;
        const result = resolveEntities(breakdown);

        // team-1 should have key1 and key1b in api_key_breakdown
        expect(Object.keys(result["team-1"].api_key_breakdown)).toEqual(["key1", "key1b"]);
        // team-2 should have key2
        expect(Object.keys(result["team-2"].api_key_breakdown)).toEqual(["key2"]);
      });
    });

    describe("getEntityBreakdown with aggregated data", () => {
      it("should produce breakdown from api_keys when entities is empty", () => {
        const result = getEntityBreakdown(aggregatedSpendData);
        expect(result.length).toBeGreaterThan(0);

        // Sorted by spend desc: team-2 (20.3) then team-1 (15.5)
        expect(result[0].metrics.spend).toBe(20.3);
        expect(result[1].metrics.spend).toBe(15.5);
      });
    });

    describe("generateDailyData with aggregated data", () => {
      it("should produce rows from api_keys when entities is empty", () => {
        const result = generateDailyData(aggregatedSpendData, "Team");
        expect(result.length).toBeGreaterThan(0);
        expect(result[0]).toHaveProperty("Date");
        expect(result[0]).toHaveProperty("Team");
      });
    });

    describe("generateDailyWithKeysData with aggregated data", () => {
      it("should produce rows from api_keys when entities is empty", () => {
        const result = generateDailyWithKeysData(aggregatedSpendData, "Team");
        expect(result.length).toBeGreaterThan(0);

        // Should have 3 key rows (key1, key1b, key2)
        expect(result).toHaveLength(3);
        const keyIds = result.map((r) => r["Key ID"]);
        expect(keyIds).toContain("key1");
        expect(keyIds).toContain("key1b");
        expect(keyIds).toContain("key2");
      });
    });

    describe("generateDailyWithModelsData with aggregated data", () => {
      it("should produce rows from api_keys when entities is empty", () => {
        const result = generateDailyWithModelsData(aggregatedSpendData, "Team");
        expect(result.length).toBeGreaterThan(0);
        expect(result[0]).toHaveProperty("Model");

        // team-1 = key1 (10.5) + key1b (5) on gpt-4; team-2 = key2 (20.3) on gpt-4.
        // Spend must aggregate per team-key, not repeat the model total per team.
        const team1 = result.find((r) => r["Team ID"] === "team-1");
        const team2 = result.find((r) => r["Team ID"] === "team-2");
        expect(team1?.["Spend ($)"]).toBe("15.5000");
        expect(team2?.["Spend ($)"]).toBe("20.3000");
      });
    });
  });

  describe("display name resolution from entity metadata", () => {
    const entityMetrics = {
      spend: 12.25,
      api_requests: 40,
      successful_requests: 39,
      failed_requests: 1,
      total_tokens: 900,
      prompt_tokens: 500,
      completion_tokens: 400,
      cache_read_input_tokens: 20,
      cache_creation_input_tokens: 10,
    };

    const makeSpendData = (entity: string, metadata?: Record<string, any>): EntitySpendData => ({
      results: [
        {
          date: "2025-04-01",
          breakdown: {
            entities: {
              [entity]: {
                metrics: entityMetrics,
                metadata,
                api_key_breakdown: {
                  key1: {
                    metrics: entityMetrics,
                    metadata: { key_alias: "prod-key" },
                  },
                },
              },
            },
          },
        },
      ],
      metadata: mockSpendData.metadata,
    });

    it("should export the user email as the entity label and keep the raw user id in the id column", () => {
      const result = generateDailyData(
        makeSpendData("user-123", { user_email: "ada@example.com", user_alias: "Ada" }),
        "User",
      );

      expect(result).toHaveLength(1);
      expect(result[0]["User"]).toBe("ada@example.com");
      expect(result[0]["User ID"]).toBe("user-123");
    });

    it("should fall back to the user alias when the user has no email", () => {
      const nullEmail = generateDailyData(
        makeSpendData("user-123", { user_email: null, user_alias: "Ada Lovelace" }),
        "User",
      );
      const missingEmail = generateDailyData(makeSpendData("user-123", { user_alias: "Ada Lovelace" }), "User");

      expect(nullEmail[0]["User"]).toBe("Ada Lovelace");
      expect(missingEmail[0]["User"]).toBe("Ada Lovelace");
    });

    it("should fall back to the raw entity key when the entity carries no metadata", () => {
      const noMetadata = generateDailyData(makeSpendData("my-tag"), "Tag");
      const emptyMetadata = generateDailyData(makeSpendData("customer-9", {}), "Customer");
      const blankNames = generateDailyData(makeSpendData("user-123", { user_email: null, user_alias: null }), "User");

      expect(noMetadata[0]["Tag"]).toBe("my-tag");
      expect(emptyMetadata[0]["Customer"]).toBe("customer-9");
      expect(blankNames[0]["User"]).toBe("user-123");
    });

    it("should prefer the team alias map over any alias in entity metadata", () => {
      const result = generateDailyData(
        makeSpendData("team-1", { team_alias: "Stale Alias", user_email: "ada@example.com" }),
        "Team",
        mockTeamAliasMap,
      );

      expect(result[0]["Team"]).toBe("Team One");
    });

    it("should use the team alias from entity metadata when the alias map has no entry for the team", () => {
      const result = generateDailyData(
        makeSpendData("team-9", { team_alias: "Team Nine", user_email: "ada@example.com" }),
        "Team",
        mockTeamAliasMap,
      );

      expect(result[0]["Team"]).toBe("Team Nine");
    });

    it("should resolve metadata.alias to the user email in getEntityBreakdown", () => {
      const withEmail = getEntityBreakdown(
        makeSpendData("user-123", { user_email: "ada@example.com", user_alias: "Ada" }),
      );
      const withoutEmail = getEntityBreakdown(makeSpendData("user-123", { user_alias: "Ada" }));

      expect(withEmail[0].metadata.alias).toBe("ada@example.com");
      expect(withEmail[0].metadata.id).toBe("user-123");
      expect(withoutEmail[0].metadata.alias).toBe("Ada");
    });

    it("should resolve the user email on every key row of the keys scope", () => {
      const spendData: EntitySpendData = {
        results: [
          {
            date: "2025-04-01",
            breakdown: {
              entities: {
                "user-123": {
                  metrics: entityMetrics,
                  metadata: { user_email: "ada@example.com", user_alias: "Ada" },
                  api_key_breakdown: {
                    key1: { metrics: entityMetrics, metadata: { key_alias: "prod-key" } },
                    key2: { metrics: entityMetrics, metadata: { key_alias: "dev-key" } },
                  },
                },
              },
            },
          },
        ],
        metadata: mockSpendData.metadata,
      };

      const result = generateDailyWithKeysData(spendData, "User");

      expect(result).toHaveLength(2);
      expect(result.map((r) => r["User"])).toEqual(["ada@example.com", "ada@example.com"]);
      expect(result.map((r) => r["User ID"])).toEqual(["user-123", "user-123"]);
      expect(result.find((r) => r["Key ID"] === "key1")?.["Key Alias"]).toBe("prod-key");
      expect(result.find((r) => r["Key ID"] === "key2")?.["Key Alias"]).toBe("dev-key");
    });

    it("should resolve each entity's own email in the models scope", () => {
      const spendData: EntitySpendData = {
        results: [
          {
            date: "2025-04-01",
            breakdown: {
              entities: {
                "user-a": {
                  metrics: entityMetrics,
                  metadata: { user_email: "ada@example.com", user_alias: "Ada" },
                  api_key_breakdown: { key1: { metrics: entityMetrics, metadata: {} } },
                },
                "user-b": {
                  metrics: entityMetrics,
                  metadata: { user_email: null, user_alias: "Grace" },
                  api_key_breakdown: { key2: { metrics: entityMetrics, metadata: {} } },
                },
              },
              models: {
                "claude-sonnet-4-5": {
                  metrics: entityMetrics,
                  api_key_breakdown: {
                    key1: { metrics: entityMetrics, metadata: {} },
                    key2: { metrics: entityMetrics, metadata: {} },
                  },
                },
              },
            },
          },
        ],
        metadata: mockSpendData.metadata,
      };

      const result = generateDailyWithModelsData(spendData, "User");

      expect(result).toHaveLength(2);
      expect(result.every((r) => r.Model === "claude-sonnet-4-5")).toBe(true);
      expect(result.find((r) => r["User ID"] === "user-a")?.["User"]).toBe("ada@example.com");
      expect(result.find((r) => r["User ID"] === "user-b")?.["User"]).toBe("Grace");
    });
  });

  describe("generateDailyWithUsersData", () => {
    it("should reconcile spend with daily_with_keys and daily per date and team", () => {
      const byUser = generateDailyWithUsersData(usersFixture, "Team");
      const byKey = generateDailyWithKeysData(usersFixture, "Team");
      const daily = generateDailyData(usersFixture, "Team");

      expect(byUser.length).toBeGreaterThan(0);
      expect(byKey.length).toBeGreaterThan(0);
      expect(daily.length).toBeGreaterThan(0);

      const sumSpend = (rows: any[]): Record<string, number> => {
        const totals: Record<string, number> = {};
        rows.forEach((r) => {
          const bucket = `${r.Date}|${r["Team ID"]}`;
          totals[bucket] = (totals[bucket] || 0) + Number(r["Spend ($)"]);
        });
        return totals;
      };

      const userTotals = sumSpend(byUser);
      const keyTotals = sumSpend(byKey);

      daily.forEach((row) => {
        const bucket = `${row.Date}|${row["Team ID"]}`;
        expect(userTotals[bucket]).toBeCloseTo(Number(row["Spend ($)"]), 4);
        expect(keyTotals[bucket]).toBeCloseTo(Number(row["Spend ($)"]), 4);
      });
    });

    it("should roll multiple keys owned by one user in a team into a single row", () => {
      const rows = generateDailyWithUsersData(usersFixture, "Team");
      const matches = rows.filter((r) => r.Date === "2025-03-01" && r["Team ID"] === "team-1" && r["User ID"] === "u1");

      expect(matches).toHaveLength(1);
      const row = matches[0];
      expect(row.Keys).toBe(2);
      expect(row["User Email"]).toBe("a@x");
      expect(row["Spend ($)"]).toBe("3.3000");
      expect(row.Requests).toBe(33);
      expect(row["Successful Requests"]).toBe(30);
      expect(row["Failed Requests"]).toBe(3);
      expect(row["Total Tokens"]).toBe(330);
      expect(row["Prompt Tokens"]).toBe(190);
      expect(row["Completion Tokens"]).toBe(140);
      expect(row["Cache Read Input Tokens"]).toBe(18);
      expect(row["Cache Creation Input Tokens"]).toBe(12);
    });

    it("should bucket keys with no owner into an Unassigned row without dropping spend", () => {
      const rows = generateDailyWithUsersData(usersFixture, "Team");
      const row = rows.find(
        (r) => r.Date === "2025-03-01" && r["Team ID"] === "team-1" && r["User ID"] === "Unassigned",
      );

      expect(row).toBeDefined();
      expect(row?.["User Email"]).toBe("-");
      expect(Number(row?.["Spend ($)"])).toBeCloseTo(4.4, 4);
    });

    it("should keep different users in the same team as separate rows", () => {
      const rows = generateDailyWithUsersData(usersFixture, "Team");
      const teamRows = rows.filter((r) => r.Date === "2025-03-01" && r["Team ID"] === "team-1");

      const u1Rows = teamRows.filter((r) => r["User ID"] === "u1");
      const u2Rows = teamRows.filter((r) => r["User ID"] === "u2");
      expect(u1Rows).toHaveLength(1);
      expect(u2Rows).toHaveLength(1);
      expect(Number(u2Rows[0]["Spend ($)"])).toBeCloseTo(3.3, 4);
    });

    it("should keep the same user in different teams as separate rows", () => {
      const rows = generateDailyWithUsersData(usersFixture, "Team");
      const u1Rows = rows.filter((r) => r.Date === "2025-03-01" && r["User ID"] === "u1");

      expect(u1Rows).toHaveLength(2);
      const team1Row = u1Rows.find((r) => r["Team ID"] === "team-1");
      const team2Row = u1Rows.find((r) => r["Team ID"] === "team-2");
      expect(team1Row?.Keys).toBe(2);
      expect(team2Row?.Keys).toBe(1);
      expect(Number(team2Row?.["Spend ($)"])).toBeCloseTo(6.6, 4);
    });

    it("should show a dash email when the user has none", () => {
      const rows = generateDailyWithUsersData(usersFixture, "Team");
      const row = rows.find((r) => r.Date === "2025-03-01" && r["Team ID"] === "team-1" && r["User ID"] === "u3");

      expect(row).toBeDefined();
      expect(row?.["User Email"]).toBe("-");
    });

    it("should still attribute a deleted key to its user", () => {
      const rows = generateDailyWithUsersData(usersFixture, "Team");
      const row = rows.find((r) => r.Date === "2025-03-01" && r["Team ID"] === "team-1" && r["User ID"] === "u3");

      expect(row).toBeDefined();
      expect(Number(row?.["Spend ($)"])).toBeCloseTo(5.5, 4);
      expect(row?.Requests).toBe(55);
    });

    it("should group rows under team_id on the aggregated endpoint shape", () => {
      const aggregatedFixture: EntitySpendData = {
        results: [
          {
            date: "2025-03-01",
            breakdown: {
              entities: {},
              api_keys: {
                kA: {
                  metrics: {
                    spend: 1.1,
                    api_requests: 11,
                    successful_requests: 10,
                    failed_requests: 1,
                    total_tokens: 110,
                    prompt_tokens: 60,
                    completion_tokens: 50,
                    cache_read_input_tokens: 6,
                    cache_creation_input_tokens: 4,
                  },
                  metadata: { team_id: "team-1", user_id: "u1", user_email: "a@x" },
                },
                kF: {
                  metrics: {
                    spend: 6.6,
                    api_requests: 66,
                    successful_requests: 64,
                    failed_requests: 2,
                    total_tokens: 660,
                    prompt_tokens: 370,
                    completion_tokens: 290,
                    cache_read_input_tokens: 36,
                    cache_creation_input_tokens: 24,
                  },
                  metadata: { team_id: "team-2", user_id: "u2" },
                },
              },
            },
          },
        ],
        metadata: usersFixture.metadata,
      };

      const rows = generateDailyWithUsersData(aggregatedFixture, "Team");

      expect(rows).toHaveLength(2);
      const team1Row = rows.find((r) => r["Team ID"] === "team-1");
      const team2Row = rows.find((r) => r["Team ID"] === "team-2");
      expect(team1Row?.["User ID"]).toBe("u1");
      expect(team2Row?.["User ID"]).toBe("u2");
      expect(Number(team1Row?.["Spend ($)"])).toBeCloseTo(1.1, 4);
      expect(Number(team2Row?.["Spend ($)"])).toBeCloseTo(6.6, 4);
    });

    it("should emit the exact column order and sort by date ascending", () => {
      const rows = generateDailyWithUsersData(usersFixture, "Team");

      expect(Object.keys(rows[0])).toEqual([
        "Date",
        "Team",
        "Team ID",
        "User ID",
        "User Email",
        "Keys",
        "Spend ($)",
        "Requests",
        "Successful Requests",
        "Failed Requests",
        "Total Tokens",
        "Prompt Tokens",
        "Completion Tokens",
        "Cache Read Input Tokens",
        "Cache Creation Input Tokens",
      ]);

      const dates = rows.map((r) => new Date(r.Date).getTime());
      for (let i = 0; i < dates.length - 1; i++) {
        expect(dates[i]).toBeLessThanOrEqual(dates[i + 1]);
      }
    });

    it("should dispatch daily_with_users through generateExportData", () => {
      expect(generateExportData(usersFixture, "daily_with_users", "Team")).toEqual(
        generateDailyWithUsersData(usersFixture, "Team"),
      );
    });

    it("should keep owners separate when entity and user ids contain underscores", () => {
      const collisionFixture: EntitySpendData = {
        results: [
          {
            date: "2025-03-01",
            breakdown: {
              entities: {
                team_1: {
                  metrics: { spend: 1, api_requests: 1, total_tokens: 10 },
                  api_key_breakdown: {
                    kX: {
                      metrics: { spend: 1, api_requests: 1, total_tokens: 10 },
                      metadata: { team_id: "team_1", user_id: "u1" },
                    },
                  },
                },
                team: {
                  metrics: { spend: 2, api_requests: 2, total_tokens: 20 },
                  api_key_breakdown: {
                    kY: {
                      metrics: { spend: 2, api_requests: 2, total_tokens: 20 },
                      metadata: { team_id: "team", user_id: "1_u1" },
                    },
                  },
                },
              },
            },
          },
        ],
        metadata: usersFixture.metadata,
      };

      const rows = generateDailyWithUsersData(collisionFixture, "Team");

      expect(rows).toHaveLength(2);
      const team1Row = rows.find((r) => r["Team ID"] === "team_1");
      expect(team1Row?.["User ID"]).toBe("u1");
      expect(team1Row?.Keys).toBe(1);
      expect(team1Row?.["Spend ($)"]).toBe("1.0000");
      const teamRow = rows.find((r) => r["Team ID"] === "team");
      expect(teamRow?.["User ID"]).toBe("1_u1");
      expect(teamRow?.Keys).toBe(1);
      expect(teamRow?.["Spend ($)"]).toBe("2.0000");
    });

    it("should leave daily and daily_with_models output without user columns", () => {
      const daily = generateDailyData(usersFixture, "Team");
      expect(daily[0]).not.toHaveProperty("User ID");

      const modelsFixture: EntitySpendData = {
        results: [
          {
            date: "2025-03-01",
            breakdown: {
              entities: {
                "team-1": {
                  metrics: {
                    spend: 1.1,
                    api_requests: 11,
                    successful_requests: 10,
                    failed_requests: 1,
                    total_tokens: 110,
                    prompt_tokens: 60,
                    completion_tokens: 50,
                  },
                  api_key_breakdown: {
                    kA: {
                      metrics: {
                        spend: 1.1,
                        api_requests: 11,
                        successful_requests: 10,
                        failed_requests: 1,
                        total_tokens: 110,
                      },
                      metadata: { team_id: "team-1", user_id: "u1", user_email: "a@x" },
                    },
                  },
                },
              },
              models: {
                "gpt-4o": {
                  metrics: { spend: 1.1, api_requests: 11, total_tokens: 110 },
                  api_key_breakdown: {
                    kA: {
                      metrics: {
                        spend: 1.1,
                        api_requests: 11,
                        successful_requests: 10,
                        failed_requests: 1,
                        total_tokens: 110,
                      },
                      metadata: {},
                    },
                  },
                },
              },
            },
          },
        ],
        metadata: usersFixture.metadata,
      };

      const modelRows = generateDailyWithModelsData(modelsFixture, "Team");
      expect(modelRows).toHaveLength(1);
      expect(Object.keys(modelRows[0])).toEqual([
        "Date",
        "Team",
        "Team ID",
        "Model",
        "Spend ($)",
        "Requests",
        "Successful",
        "Failed",
        "Total Tokens",
        "Prompt Tokens",
        "Completion Tokens",
        "Cache Read Input Tokens",
        "Cache Creation Input Tokens",
      ]);
    });
  });

  describe("handleServerExport", () => {
    beforeEach(() => {
      document.body.innerHTML = "";
      window.URL.createObjectURL = vi.fn(() => "blob:mock-url");
      window.URL.revokeObjectURL = vi.fn();
    });

    afterEach(() => {
      vi.restoreAllMocks();
    });

    it("passes the chosen scope and format to the server export and downloads the returned blob", async () => {
      const serverBlob = new Blob(["payload"], { type: "text/csv" });
      const serverExport = vi.fn(async () => serverBlob);
      const createObjectURLSpy = vi.spyOn(window.URL, "createObjectURL");
      const appendChildSpy = vi.spyOn(document.body, "appendChild");

      await handleServerExport(serverExport, "daily_with_keys", "team", "csv");

      expect(serverExport).toHaveBeenCalledWith("daily_with_keys", "csv");
      expect(createObjectURLSpy).toHaveBeenCalledWith(serverBlob);
      const attached = appendChildSpy.mock.calls[0][0] as HTMLAnchorElement;
      const today = new Date().toISOString().split("T")[0];
      expect(attached.download).toBe(`team_usage_daily_with_keys_${today}.csv`);
    });

    it("lets a server failure propagate so the modal can toast it instead of downloading nothing", async () => {
      const serverExport = vi.fn(async () => {
        throw new Error("upstream 500");
      });

      await expect(handleServerExport(serverExport, "daily", "team", "json")).rejects.toThrow("upstream 500");
      expect(document.body.querySelector("a")).toBeNull();
    });
  });
});
