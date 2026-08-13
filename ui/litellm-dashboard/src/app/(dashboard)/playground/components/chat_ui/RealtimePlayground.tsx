"use client";

import { Loader2, Mic, MicOff, Phone, PhoneOff, Volume2 } from "lucide-react";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { getProxyBaseUrl } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/cva.config";
import ChatComposer from "./ChatComposer";
import { OPEN_AI_VOICE_SELECT_OPTIONS, type OpenAIVoice } from "./chatConstants";

interface RealtimeMessage {
  role: "user" | "assistant" | "system" | "status";
  content: string;
  timestamp: Date;
}

interface RealtimePlaygroundProps {
  accessToken: string;
  selectedModel: string;
  customProxyBaseUrl?: string;
  selectedGuardrails?: string[];
}

const RealtimePlayground: React.FC<RealtimePlaygroundProps> = ({
  accessToken,
  selectedModel,
  customProxyBaseUrl,
  selectedGuardrails,
}) => {
  const [messages, setMessages] = useState<RealtimeMessage[]>([]);
  const [inputText, setInputText] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [selectedVoice, setSelectedVoice] = useState<OpenAIVoice>("alloy");
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const nextPlayTimeRef = useRef(0);
  const selectedVoiceRef = useRef(selectedVoice);

  useEffect(() => {
    selectedVoiceRef.current = selectedVoice;
  }, [selectedVoice]);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const addMessage = useCallback((role: RealtimeMessage["role"], content: string) => {
    setMessages((prev) => [...prev, { role, content, timestamp: new Date() }]);
  }, []);

  const appendAssistantText = useCallback((text: string) => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.role === "assistant") {
        return [...prev.slice(0, -1), { ...last, content: last.content + text }];
      }
      return [...prev, { role: "assistant", content: text, timestamp: new Date() }];
    });
  }, []);

  const playAudioChunk = useCallback((base64Audio: string) => {
    const raw = atob(base64Audio);
    const bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    const pcm16 = new Int16Array(bytes.buffer);
    const float32 = new Float32Array(pcm16.length);
    for (let i = 0; i < pcm16.length; i++) float32[i] = pcm16[i] / 32768;

    const ctx = audioContextRef.current;
    if (!ctx) return;

    const buffer = ctx.createBuffer(1, float32.length, 24000);
    buffer.getChannelData(0).set(float32);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    const now = ctx.currentTime;
    const startTime = Math.max(now, nextPlayTimeRef.current);
    source.start(startTime);
    nextPlayTimeRef.current = startTime + buffer.duration;
  }, []);

  const stopRecording = useCallback(() => {
    processorRef.current?.disconnect();
    processorRef.current = null;
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    mediaStreamRef.current = null;
    setIsRecording(false);
  }, []);

  const connect = useCallback(async () => {
    if (wsRef.current) return;
    if (!selectedModel) {
      addMessage("status", "Please select a model first");
      return;
    }
    setIsConnecting(true);

    try {
      audioContextRef.current = new AudioContext({ sampleRate: 24000 });

      const baseUrl = customProxyBaseUrl || getProxyBaseUrl();
      const wsBase = baseUrl.replace(/^http/, "ws");
      let url = `${wsBase}/v1/realtime?model=${encodeURIComponent(selectedModel)}`;
      if (selectedGuardrails && selectedGuardrails.length > 0) {
        url += `&guardrails=${encodeURIComponent(selectedGuardrails.join(","))}`;
      }

      const ws = new WebSocket(url, ["realtime", `openai-insecure-api-key.${accessToken}`]);

      ws.onopen = () => {
        setIsConnected(true);
        setIsConnecting(false);
        addMessage("status", "Connected to realtime API");
      };

      ws.onmessage = async (event) => {
        try {
          let raw = event.data;
          if (raw instanceof Blob) {
            raw = await raw.text();
          } else if (raw instanceof ArrayBuffer) {
            raw = new TextDecoder().decode(raw);
          }
          const data = JSON.parse(raw);
          const type = data.type as string;

          if (type === "session.created") {
            ws.send(
              JSON.stringify({
                type: "session.update",
                session: {
                  type: "realtime",
                  modalities: ["text", "audio"],
                  voice: selectedVoiceRef.current,
                  input_audio_format: "pcm16",
                  output_audio_format: "pcm16",
                  input_audio_transcription: { model: "gpt-4o-mini-transcribe" },
                  turn_detection: null,
                },
              }),
            );
          } else if (type === "response.output_audio.delta" || type === "response.audio.delta") {
            if (data.delta) playAudioChunk(data.delta);
          } else if (
            type === "response.output_text.delta" ||
            type === "response.output_audio_transcript.delta" ||
            type === "response.audio_transcript.delta" ||
            type === "response.text.delta"
          ) {
            if (data.delta) appendAssistantText(data.delta);
          } else if (type === "conversation.item.input_audio_transcription.completed") {
            if (data.transcript) addMessage("user", data.transcript);
          } else if (type === "response.done") {
            setMessages((prev) => {
              const last = prev[prev.length - 1];
              if (last && last.role === "assistant" && last.content) return prev;
              const output = data.response?.output || [];
              const texts: string[] = [];
              for (const item of output) {
                for (const part of item.content || []) {
                  const t = part.text || part.transcript;
                  if (t) texts.push(t);
                }
              }
              if (texts.length > 0) {
                return [...prev, { role: "assistant", content: texts.join(""), timestamp: new Date() }];
              }
              return prev;
            });
          } else if (type === "error") {
            addMessage("status", `Error: ${data.error?.message || JSON.stringify(data.error)}`);
          }
        } catch {
        }
      };

      ws.onerror = () => {
        addMessage("status", "WebSocket error");
        setIsConnected(false);
        setIsConnecting(false);
      };

      ws.onclose = () => {
        addMessage("status", "Disconnected");
        setIsConnected(false);
        setIsConnecting(false);
        wsRef.current = null;
      };

      wsRef.current = ws;
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      addMessage("status", `Connection failed: ${message}`);
      setIsConnecting(false);
    }
  }, [accessToken, selectedModel, customProxyBaseUrl, selectedGuardrails, addMessage, appendAssistantText, playAudioChunk]);

  const disconnect = useCallback(() => {
    stopRecording();
    wsRef.current?.close();
    wsRef.current = null;
    audioContextRef.current?.close();
    audioContextRef.current = null;
    nextPlayTimeRef.current = 0;
    setIsConnected(false);
  }, [stopRecording]);

  const startRecording = useCallback(async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    wsRef.current.send(
      JSON.stringify({
        type: "session.update",
        session: {
          type: "realtime",
          modalities: ["text", "audio"],
          voice: selectedVoice,
          input_audio_format: "pcm16",
          output_audio_format: "pcm16",
          input_audio_transcription: { model: "gpt-4o-mini-transcribe" },
          turn_detection: { type: "server_vad" },
        },
      }),
    );

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const ctx = audioContextRef.current || new AudioContext({ sampleRate: 24000 });
      audioContextRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        const input = e.inputBuffer.getChannelData(0);

        const sampleRate = ctx.sampleRate;
        const targetRate = 24000;
        let samples: Float32Array;
        if (sampleRate !== targetRate) {
          const ratio = sampleRate / targetRate;
          const newLength = Math.round(input.length / ratio);
          samples = new Float32Array(newLength);
          for (let i = 0; i < newLength; i++) {
            samples[i] = input[Math.round(i * ratio)] || 0;
          }
        } else {
          samples = input;
        }

        const pcm16 = new Int16Array(samples.length);
        for (let i = 0; i < samples.length; i++) {
          const s = Math.max(-1, Math.min(1, samples[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }

        const bytes = new Uint8Array(pcm16.buffer);
        let binary = "";
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        const b64 = btoa(binary);

        wsRef.current.send(JSON.stringify({ type: "input_audio_buffer.append", audio: b64 }));
      };

      source.connect(processor);
      processor.connect(ctx.destination);
      setIsRecording(true);
      addMessage("status", "Listening...");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      addMessage("status", `Microphone error: ${message}`);
    }
  }, [addMessage, selectedVoice]);

  const sendTextMessage = useCallback(() => {
    if (!inputText.trim() || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    stopRecording();

    const text = inputText.trim();
    addMessage("user", text);
    setInputText("");

    wsRef.current.send(
      JSON.stringify({
        type: "session.update",
        session: {
          type: "realtime",
          modalities: ["text", "audio"],
          voice: selectedVoiceRef.current,
          input_audio_format: "pcm16",
          output_audio_format: "pcm16",
          input_audio_transcription: { model: "gpt-4o-mini-transcribe" },
          turn_detection: null,
        },
      }),
    );

    wsRef.current.send(
      JSON.stringify({
        type: "conversation.item.create",
        item: {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text }],
        },
      }),
    );
    wsRef.current.send(JSON.stringify({ type: "response.create" }));
  }, [inputText, addMessage, stopRecording]);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
      audioContextRef.current?.close();
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const voiceLabel =
    OPEN_AI_VOICE_SELECT_OPTIONS.find((voice) => voice.value === selectedVoice)?.label ?? selectedVoice;

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-gray-200 bg-gray-50 px-4 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <Volume2 className="size-5 shrink-0 text-blue-500" aria-hidden="true" />
          <div className="min-w-0">
            <p className="font-semibold text-gray-800">Realtime Voice Chat</p>
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <span
                className={cn("inline-block size-2 rounded-full", isConnected ? "bg-green-500" : "bg-gray-300")}
                aria-hidden="true"
              />
              {!isConnected ? (isConnecting ? "Connecting..." : "Disconnected") : "Connected"}
              {selectedModel ? <span className="truncate">· {selectedModel}</span> : null}
            </div>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Select
            value={selectedVoice}
            onValueChange={(value) => setSelectedVoice(value as OpenAIVoice)}
            disabled={isConnected || isConnecting}
          >
            <SelectTrigger className="w-[220px]" size="sm" aria-label="Realtime voice">
              <SelectValue>{voiceLabel}</SelectValue>
            </SelectTrigger>
            <SelectContent>
              {OPEN_AI_VOICE_SELECT_OPTIONS.map((voice) => (
                <SelectItem key={voice.value} value={voice.value}>
                  {voice.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {!isConnected ? (
            <Button type="button" size="sm" onClick={() => void connect()} disabled={isConnecting || !selectedModel}>
              {isConnecting ? <Loader2 className="size-3.5 animate-spin" /> : <Phone className="size-3.5" />}
              Connect
            </Button>
          ) : (
            <Button type="button" size="sm" variant="destructive" onClick={disconnect}>
              <PhoneOff className="size-3.5" />
              Disconnect
            </Button>
          )}
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
        {messages.length === 0 && !isConnected && (
          <div className="flex h-full flex-col items-center justify-center gap-3 text-gray-400">
            <Volume2 className="size-12" aria-hidden="true" />
            <p className="text-lg text-gray-500">Realtime Voice Playground</p>
            <p className="max-w-md text-center text-sm text-gray-400">
              Select a realtime model, pick a voice, then click <b>Connect</b>. You can speak with the mic or type
              messages. The model responds with voice and text.
            </p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={`${msg.timestamp.toISOString()}-${i}`}
            className={cn(
              "flex",
              msg.role === "user" ? "justify-end" : msg.role === "status" ? "justify-center" : "justify-start",
            )}
          >
            {msg.role === "status" ? (
              <div className="px-3 py-1 text-xs italic text-gray-400">{msg.content}</div>
            ) : (
              <div
                className={cn(
                  "max-w-[75%] rounded-2xl px-4 py-2.5",
                  msg.role === "user"
                    ? "rounded-br-md bg-blue-500 text-white"
                    : "rounded-bl-md bg-gray-100 text-gray-800",
                )}
              >
                <div className="mb-0.5 text-xs font-medium opacity-70">{msg.role === "user" ? "You" : "AI"}</div>
                <div className="whitespace-pre-wrap text-sm">{msg.content}</div>
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {isConnected && (
        <div className="shrink-0 border-t border-gray-200 bg-white p-3 sm:p-4">
          {isRecording && (
            <div className="mb-3 flex items-center gap-2 text-xs text-red-500">
              <span className="inline-block size-2 animate-pulse rounded-full bg-red-500" aria-hidden="true" />
              Listening. Speak into your microphone. Server VAD will detect when you stop.
            </div>
          )}
          <ChatComposer
            value={inputText}
            onChange={setInputText}
            onSubmit={sendTextMessage}
            placeholder="Type a message or use the mic..."
            submitDisabled={!inputText.trim()}
            tools={
              <Tooltip>
                <TooltipTrigger
                  render={
                    <Button
                      type="button"
                      variant={isRecording ? "destructive" : "ghost"}
                      size="icon-sm"
                      className={cn("size-8 rounded-lg border border-border/40", isRecording && "animate-pulse")}
                      aria-label={isRecording ? "Stop recording" : "Start recording"}
                      onClick={() => {
                        if (isRecording) {
                          stopRecording();
                        } else {
                          void startRecording();
                        }
                      }}
                    />
                  }
                >
                  {isRecording ? <MicOff className="size-4" /> : <Mic className="size-4" />}
                </TooltipTrigger>
                <TooltipContent>{isRecording ? "Stop recording" : "Start recording"}</TooltipContent>
              </Tooltip>
            }
          />
        </div>
      )}
    </div>
  );
};

export default RealtimePlayground;
