import type { ChatAttachment, Conversation } from "./types";

const DATABASE_NAME = "litellm_chat_media_v1";
const STORE_NAME = "images";
let databasePromise: Promise<IDBDatabase> | undefined;
const imageWrites = new Map<string, Promise<void>>();

function openDatabase(): Promise<IDBDatabase> {
  if (!databasePromise) {
    databasePromise = new Promise<IDBDatabase>((resolve, reject) => {
      const request = indexedDB.open(DATABASE_NAME, 1);
      request.onupgradeneeded = () => {
        request.result.createObjectStore(STORE_NAME, { keyPath: ["userId", "id"] });
      };
      request.onsuccess = () => {
        const database = request.result;
        database.onversionchange = () => {
          database.close();
          databasePromise = undefined;
          imageWrites.clear();
        };
        resolve(database);
      };
      request.onerror = () => reject(request.error ?? new Error("Attachment storage is unavailable."));
      request.onblocked = () => reject(new Error("Attachment storage is blocked by another browser tab."));
    }).catch((error: unknown) => {
      databasePromise = undefined;
      throw error;
    });
  }
  return databasePromise;
}

export function getConversationImages(conversations: Conversation[]): ChatAttachment[] {
  return conversations.flatMap((conversation) =>
    conversation.messages.flatMap((message) => [...(message.attachments ?? []), ...(message.generatedImages ?? [])]),
  );
}

export function persistChatImages(userId: string, images: ChatAttachment[]): Promise<void> {
  return Promise.all(
    images
      .filter((image) => image.url.startsWith("data:"))
      .map((image) => {
        const key = JSON.stringify([userId, image.id]);
        let write = imageWrites.get(key);
        if (!write) {
          write = openDatabase()
            .then(
              (database) =>
                new Promise<void>((resolve, reject) => {
                  const transaction = database.transaction(STORE_NAME, "readwrite");
                  transaction.objectStore(STORE_NAME).put({ userId, id: image.id, url: image.url });
                  transaction.oncomplete = () => resolve();
                  transaction.onerror = () => reject(transaction.error ?? new Error("Could not save the attachment."));
                  transaction.onabort = () =>
                    reject(transaction.error ?? new Error("Saving the attachment was canceled."));
                }),
            )
            .catch((error: unknown) => {
              imageWrites.delete(key);
              throw error;
            });
          imageWrites.set(key, write);
        }
        return write;
      }),
  ).then(() => undefined);
}

export async function loadChatImages(userId: string, ids: string[]): Promise<Map<string, string>> {
  if (!ids.length) return new Map();
  const database = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, "readonly");
    const loaded = new Map<string, string>();
    for (const id of new Set(ids)) {
      const request = transaction.objectStore(STORE_NAME).get([userId, id]);
      request.onsuccess = () => {
        if (typeof request.result?.url === "string") loaded.set(id, request.result.url);
      };
    }
    transaction.oncomplete = () => resolve(loaded);
    transaction.onerror = () => reject(transaction.error ?? new Error("Could not restore saved attachments."));
    transaction.onabort = () => reject(transaction.error ?? new Error("Restoring saved attachments was canceled."));
  });
}

export async function deleteChatImages(userId: string, ids: string[]): Promise<void> {
  if (!ids.length) return;
  for (const id of ids) imageWrites.delete(JSON.stringify([userId, id]));
  const database = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, "readwrite");
    for (const id of ids) transaction.objectStore(STORE_NAME).delete([userId, id]);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error ?? new Error("Could not delete saved attachments."));
    transaction.onabort = () => reject(transaction.error ?? new Error("Deleting saved attachments was canceled."));
  });
}

export function withoutImageData(conversations: Conversation[]): Conversation[] {
  const stripData = <T extends ChatAttachment>(images: T[] | undefined) =>
    images?.map((image) => (image.url.startsWith("data:") ? { ...image, url: "" } : image));
  return conversations.map((conversation) => ({
    ...conversation,
    messages: conversation.messages.map((message) => ({
      ...message,
      ...(message.attachments ? { attachments: stripData(message.attachments) } : {}),
      ...(message.generatedImages ? { generatedImages: stripData(message.generatedImages) } : {}),
    })),
  }));
}

export function withRestoredImages(conversations: Conversation[], images: Map<string, string>): Conversation[] {
  const restore = <T extends ChatAttachment>(attachments: T[]) =>
    attachments.map((image) => (!image.url && images.has(image.id) ? { ...image, url: images.get(image.id)! } : image));
  return conversations.map((conversation) => ({
    ...conversation,
    messages: conversation.messages.map((message) => ({
      ...message,
      ...(message.attachments ? { attachments: restore(message.attachments) } : {}),
      ...(message.generatedImages ? { generatedImages: restore(message.generatedImages) } : {}),
    })),
  }));
}
