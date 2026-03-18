import { useState } from "react";

interface UseFileUploadsReturn {
  // Image edits endpoint
  uploadedImages: File[];
  imagePreviewUrls: string[];
  handleImageUpload: (file: File) => false;
  handleRemoveImage: (index: number) => void;
  handleRemoveAllImages: () => void;
  // Responses endpoint
  responsesUploadedImage: File | null;
  responsesImagePreviewUrl: string | null;
  handleResponsesImageUpload: (file: File) => false;
  handleRemoveResponsesImage: () => void;
  // Chat endpoint
  chatUploadedImage: File | null;
  chatImagePreviewUrl: string | null;
  handleChatImageUpload: (file: File) => false;
  handleRemoveChatImage: () => void;
  // Audio transcription endpoint
  uploadedAudio: File | null;
  handleAudioUpload: (file: File) => false;
  handleRemoveAudio: () => void;
  /** Clears all file uploads across all endpoint types */
  clearAllUploads: () => void;
}

function useFileUploads(): UseFileUploadsReturn {
  // Image edits
  const [uploadedImages, setUploadedImages] = useState<File[]>([]);
  const [imagePreviewUrls, setImagePreviewUrls] = useState<string[]>([]);

  // Responses
  const [responsesUploadedImage, setResponsesUploadedImage] = useState<File | null>(null);
  const [responsesImagePreviewUrl, setResponsesImagePreviewUrl] = useState<string | null>(null);

  // Chat
  const [chatUploadedImage, setChatUploadedImage] = useState<File | null>(null);
  const [chatImagePreviewUrl, setChatImagePreviewUrl] = useState<string | null>(null);

  // Audio
  const [uploadedAudio, setUploadedAudio] = useState<File | null>(null);

  // --- Image edits handlers ---

  const handleImageUpload = (file: File): false => {
    setUploadedImages((prev) => [...prev, file]);
    const previewUrl = URL.createObjectURL(file);
    setImagePreviewUrls((prev) => [...prev, previewUrl]);
    return false; // Prevent default upload behavior
  };

  const handleRemoveImage = (index: number) => {
    if (imagePreviewUrls[index]) {
      URL.revokeObjectURL(imagePreviewUrls[index]);
    }
    setUploadedImages((prev) => prev.filter((_, i) => i !== index));
    setImagePreviewUrls((prev) => prev.filter((_, i) => i !== index));
  };

  const handleRemoveAllImages = () => {
    imagePreviewUrls.forEach((url) => URL.revokeObjectURL(url));
    setUploadedImages([]);
    setImagePreviewUrls([]);
  };

  // --- Responses handlers ---

  const handleResponsesImageUpload = (file: File): false => {
    setResponsesUploadedImage(file);
    const previewUrl = URL.createObjectURL(file);
    setResponsesImagePreviewUrl(previewUrl);
    return false;
  };

  const handleRemoveResponsesImage = () => {
    if (responsesImagePreviewUrl) {
      URL.revokeObjectURL(responsesImagePreviewUrl);
    }
    setResponsesUploadedImage(null);
    setResponsesImagePreviewUrl(null);
  };

  // --- Chat handlers ---

  const handleChatImageUpload = (file: File): false => {
    setChatUploadedImage(file);
    const previewUrl = URL.createObjectURL(file);
    setChatImagePreviewUrl(previewUrl);
    return false;
  };

  const handleRemoveChatImage = () => {
    if (chatImagePreviewUrl) {
      URL.revokeObjectURL(chatImagePreviewUrl);
    }
    setChatUploadedImage(null);
    setChatImagePreviewUrl(null);
  };

  // --- Audio handlers ---

  const handleAudioUpload = (file: File): false => {
    setUploadedAudio(file);
    return false;
  };

  const handleRemoveAudio = () => {
    setUploadedAudio(null);
  };

  // --- Clear all ---

  const clearAllUploads = () => {
    handleRemoveAllImages();
    handleRemoveResponsesImage();
    handleRemoveChatImage();
    handleRemoveAudio();
  };

  return {
    uploadedImages,
    imagePreviewUrls,
    handleImageUpload,
    handleRemoveImage,
    handleRemoveAllImages,
    responsesUploadedImage,
    responsesImagePreviewUrl,
    handleResponsesImageUpload,
    handleRemoveResponsesImage,
    chatUploadedImage,
    chatImagePreviewUrl,
    handleChatImageUpload,
    handleRemoveChatImage,
    uploadedAudio,
    handleAudioUpload,
    handleRemoveAudio,
    clearAllUploads,
  };
}

export default useFileUploads;
