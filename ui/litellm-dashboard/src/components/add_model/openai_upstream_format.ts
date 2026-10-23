export enum OpenAIUpstreamFormat {
  Default = "default",
  SystemOne = "systemone",
  Decisions = "decisions",
}

export const isOpenAIUpstreamModel = (model: string | undefined, provider: string | undefined): boolean =>
  provider ? provider === "openai" : model?.startsWith("openai/") === true;

export const parseOpenAIUpstreamFormat = (value: unknown): OpenAIUpstreamFormat => {
  switch (value) {
    case OpenAIUpstreamFormat.SystemOne:
      return OpenAIUpstreamFormat.SystemOne;
    case OpenAIUpstreamFormat.Decisions:
      return OpenAIUpstreamFormat.Decisions;
    default:
      return OpenAIUpstreamFormat.Default;
  }
};

export const parseOpenAIUpstreamModel = (model: string): { model: string; format: OpenAIUpstreamFormat } => {
  for (const format of [OpenAIUpstreamFormat.SystemOne, OpenAIUpstreamFormat.Decisions]) {
    const prefix = `openai/${format}/`;
    if (model.startsWith(prefix)) {
      return { model: `openai/${model.slice(prefix.length)}`, format };
    }
  }
  return { model, format: OpenAIUpstreamFormat.Default };
};

export const serializeOpenAIUpstreamModel = (model: string, format: OpenAIUpstreamFormat): string => {
  if (format === OpenAIUpstreamFormat.Default) return model;
  const upstreamModel = model.startsWith("openai/") ? model.slice("openai/".length) : model;
  return `openai/${format}/${upstreamModel}`;
};
