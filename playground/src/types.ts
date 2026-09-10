export type PlaygroundFile = {
  languageId: 'rust' | 'toml';
  path: string;
  source: string;
  target: string;
  uri: string;
};

export type PlaygroundGuide = {
  path: string;
  source: string;
  target: string;
};

export type PlaygroundExample = {
  files: PlaygroundFile[];
  id: string;
  title: string;
};

export type PlaygroundInfo = {
  examples: PlaygroundExample[];
  guide: PlaygroundGuide;
  rootUri: string;
  revision: string;
};

export type RunResult = {
  success: boolean;
  output: string;
};

export type StoredFiles = Record<string, string>;
