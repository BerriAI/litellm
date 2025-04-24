export class UiError extends Error {
  constructor(message: string) {
    super(message);
    Object.setPrototypeOf(this, UiError.prototype)
  }
}