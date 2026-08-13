import { message as staticMessage } from "antd";
import type { MessageInstance } from "antd/es/message/interface";

let messageInstance: MessageInstance | null = null;

export const setMessageInstance = (instance: MessageInstance) => {
  messageInstance = instance;
};

const getMessageApi = () => messageInstance || staticMessage;

const MessageManager = {
  success(content: string, duration?: number) {
    getMessageApi().success(content, duration);
  },

  error(content: string, duration?: number) {
    getMessageApi().error(content, duration);
  },

  warning(content: string, duration?: number) {
    getMessageApi().warning(content, duration);
  },

  info(content: string, duration?: number) {
    getMessageApi().info(content, duration);
  },

  loading(content: string, duration?: number) {
    return getMessageApi().loading(content, duration);
  },

  destroy() {
    getMessageApi().destroy();
  },
};

export default MessageManager;
