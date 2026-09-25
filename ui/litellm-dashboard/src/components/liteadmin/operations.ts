export interface LiteAdminAction {
  id: string;
  name: string;
  title: string;
  arguments: Record<string, unknown>;
  destructive: boolean;
}

export type ActionResult = { status: "completed"; key?: string } | { status: "unknown"; message: string };

export interface OperationContext {
  accessToken: string;
  signal: AbortSignal;
  assertCurrent: () => void;
  confirm: (action: LiteAdminAction) => Promise<boolean>;
  onResult: (action: LiteAdminAction, result: ActionResult) => void;
}
