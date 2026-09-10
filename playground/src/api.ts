import type { PlaygroundInfo, RunResult, StoredFiles } from './types';

const jsonRequest = async <T>(input: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(input, init);
  if (!response.ok) {
    throw new Error(`${init?.method ?? 'GET'} ${input} failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
};

export const loadInfo = () => jsonRequest<PlaygroundInfo>('/api/info');

export const runCode = (exampleId: string, files: StoredFiles) =>
  jsonRequest<RunResult>('/api/run', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ exampleId, files }),
  });

export const persistFile = (target: string, source: string) =>
  jsonRequest<unknown>('/api/file', {
    method: 'PUT',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ target, source }),
  });
