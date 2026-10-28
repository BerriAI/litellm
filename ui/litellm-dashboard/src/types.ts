export type Setter<T> = (newValueOrUpdater: T | ((previousValue: T) => T)) => void;

// We'll get the email event types from the endpoint
export type EmailEvent = string;
