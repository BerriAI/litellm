export class LanguageModelTextPart {
  constructor(readonly value: string) {}
}

export class LanguageModelToolCallPart {
  constructor(
    readonly callId: string,
    readonly name: string,
    readonly input: object,
  ) {}
}

export const LanguageModelChatToolMode = { Auto: 1, Required: 2 } as const;

type Listener<T> = (value: T) => void;

interface Subscription {
  dispose(): void;
}

export class EventEmitter<T> {
  private readonly listeners: Listener<T>[] = [];

  readonly event = (listener: Listener<T>): Subscription => {
    this.listeners.push(listener);
    return {
      dispose: () => {
        const index = this.listeners.indexOf(listener);
        if (index >= 0) {
          this.listeners.splice(index, 1);
        }
      },
    };
  };

  fire(value: T): void {
    [...this.listeners].forEach((listener) => listener(value));
  }

  dispose(): void {
    this.listeners.splice(0);
  }
}

export interface MockCancellationToken {
  readonly isCancellationRequested: boolean;
  onCancellationRequested(listener: Listener<void>): Subscription;
}

export class CancellationTokenSource {
  private readonly emitter = new EventEmitter<void>();
  private cancelled = false;
  private disposed = 0;
  readonly token: MockCancellationToken;

  constructor() {
    const source = this;
    this.token = {
      get isCancellationRequested(): boolean {
        return source.cancelled;
      },
      onCancellationRequested: (listener) => {
        const subscription = source.emitter.event(listener);
        return {
          dispose: () => {
            source.disposed += 1;
            subscription.dispose();
          },
        };
      },
    };
  }

  get disposedListeners(): number {
    return this.disposed;
  }

  cancel(): void {
    this.cancelled = true;
    this.emitter.fire();
  }
}
