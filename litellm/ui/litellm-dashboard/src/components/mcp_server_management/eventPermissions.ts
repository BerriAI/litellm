export type EventPermissions = Readonly<Record<string, readonly string[]>>;

export function withServerEvents(
  current: EventPermissions,
  serverId: string,
  selected: readonly string[],
): Record<string, string[]> {
  const others = Object.fromEntries(
    Object.entries(current)
      .filter(([key]) => key !== serverId)
      .map(([key, events]) => [key, [...events]]),
  );
  return selected.length === 0 ? others : { ...others, [serverId]: [...selected] };
}

export function sameEventPermissions(left: EventPermissions, right: EventPermissions): boolean {
  const normalize = (value: EventPermissions) =>
    JSON.stringify(
      Object.entries(value)
        .map(([key, events]) => [key, [...events].sort()] as const)
        .sort(([a], [b]) => a.localeCompare(b)),
    );
  return normalize(left) === normalize(right);
}

export interface EventServer {
  serverId: string;
  label: string;
}

export function eventServers(
  servers: readonly { server_id: string; server_name?: string | null; alias?: string | null; events?: { enabled: boolean } | null }[],
): EventServer[] {
  return servers
    .filter((server) => server.events?.enabled === true)
    .map((server) => ({ serverId: server.server_id, label: server.server_name || server.alias || server.server_id }));
}
