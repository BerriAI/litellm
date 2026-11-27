import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { useTranslation } from "react-i18next";
import { z } from "zod";

const paramsSchema = z.object({
  enabled: z
    .boolean()
    .nullish()
    .transform((value) => value ?? undefined),
  similarity_threshold: z
    .number()
    .min(0)
    .max(1)
    .nullish()
    .transform((value) => value ?? undefined),
  ttl: z
    .number()
    .int()
    .positive()
    .nullish()
    .transform((value) => value ?? undefined),
});

type CacheParams = Partial<z.infer<typeof paramsSchema>>;

const metadataSchema = z.record(z.string(), z.unknown());

function readMetadata(value: string | null | undefined): Record<string, unknown> | undefined {
  try {
    const parsed = metadataSchema.safeParse(value?.trim() ? JSON.parse(value) : {});
    return parsed.success ? parsed.data : undefined;
  } catch {
    return undefined;
  }
}

export function SemanticCacheJsonControls({
  value,
  onChange,
}: {
  value: string | null | undefined;
  onChange: (value: string) => void;
}) {
  const metadata = readMetadata(value);
  return (
    <SemanticCacheControls
      value={metadata?.cache_params}
      disabled={metadata === undefined}
      onChange={(cacheParams) => {
        const { cache_params: _previous, ...rest } = metadata ?? {};
        onChange(JSON.stringify(cacheParams ? { ...rest, cache_params: cacheParams } : rest, null, 2));
      }}
    />
  );
}

export function SemanticCacheControls({
  value,
  onChange,
  disabled,
}: {
  value: unknown;
  onChange: (value: CacheParams | undefined) => void;
  disabled?: boolean;
}) {
  const { t } = useTranslation();
  const { data: settings } = useUISettings({ staleTime: 0 });
  const result = paramsSchema.safeParse(value);
  const params: CacheParams = result.success ? result.data : {};
  const update = (next: CacheParams) =>
    onChange(Object.values(next).some((field) => field !== undefined) ? next : undefined);
  return (
    <fieldset disabled={disabled} className="mb-4 grid gap-2">
      <legend className="font-medium">{t("shared.semanticCaching.title")}</legend>
      <label>
        {t("shared.semanticCaching.mode")}
        <select
          className="ml-2 rounded border px-2 py-1"
          value={params.enabled === undefined ? "" : String(params.enabled)}
          onChange={(event) => {
            const choice = event.currentTarget.value;
            update({ ...params, enabled: choice === "" ? undefined : choice === "true" });
          }}
        >
          <option value="">{t("shared.semanticCaching.inherit")}</option>
          <option value="true">{t("shared.semanticCaching.enabled")}</option>
          <option value="false">{t("shared.semanticCaching.disabled")}</option>
        </select>
      </label>
      {settings?.supports_similarity_threshold === true && (
        <label>
          {t("shared.semanticCaching.threshold")}
          <input
            className="ml-2 rounded border px-2 py-1"
            type="number"
            min={0}
            max={1}
            step="any"
            value={params.similarity_threshold ?? ""}
            onChange={(event) => {
              const input = event.currentTarget;
              if (input.validity.valid) {
                update({ ...params, similarity_threshold: input.value === "" ? undefined : input.valueAsNumber });
              }
            }}
          />
        </label>
      )}
      <label>
        {t("shared.semanticCaching.ttl")}
        <input
          className="ml-2 rounded border px-2 py-1"
          type="number"
          min={1}
          step={1}
          value={params.ttl ?? ""}
          onChange={(event) => {
            const input = event.currentTarget;
            if (input.validity.valid) {
              update({ ...params, ttl: input.value === "" ? undefined : input.valueAsNumber });
            }
          }}
        />
      </label>
    </fieldset>
  );
}
