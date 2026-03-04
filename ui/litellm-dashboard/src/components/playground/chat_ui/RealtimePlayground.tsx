"use client";

import {
  AudioMutedOutlined,
  AudioOutlined,
  CloseCircleOutlined,
  SendOutlined,
  SoundOutlined,
} from "@ant-design/icons";
import { Button, Input, Select, Typography } from "antd";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { getProxyBaseUrl } from "../../networking";
import { OPEN_AI_VOICE_SELECT_OPTIONS } from "./chatConstants";

const { Text } = Typography;

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
  const [selectedVoice, setSelectedVoice] = useState("alloy");
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const playbackQueueRef = useRef<ArrayBuffer[]>([]);
  const isPlayingRef = useRef(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const nextPlayTimeRef = useRef(0);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const addMessage = useCallback(
    (role: RealtimeMessage["role"], content: string) => {
      setMessages((prev) => [...prev, { role, content, timestamp: new Date() }]);
    },
    []
  );

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
          const type = data.type;

          if (type === "session.created") {
            ws.send(
              JSON.stringify({
                type: "session.update",
                session: {
                  modalities: ["text", "audio"],
                  voice: selectedVoice,
                  input_audio_format: "pcm16",
                  output_audio_format: "pcm16",
                  input_audio_transcription: { model: "gpt-4o-mini-transcribe" },
                  turn_detection: null,
                },
              })
            );
          } else if (type === "session.updated") {
            // session configured
          } else if (type === "response.audio.delta") {
            if (data.delta) playAudioChunk(data.delta);
          } else if (type === "response.audio_transcript.delta" || type === "response.text.delta") {
            if (data.delta) appendAssistantText(data.delta);
          } else if (
            type === "conversation.item.input_audio_transcription.completed"
          ) {
            if (data.transcript) addMessage("user", data.transcript);
          } else if (type === "response.done") {
            // Ensure we have the full text if deltas were missed
            setMessages((prev) => {
              const last = prev[prev.length - 1];
              if (last && last.role === "assistant" && last.content) return prev;
              // No assistant message yet — extract from response.done
              const output = data.response?.output || [];
              const texts: string[] = [];
              for (const item of output) {
                for (const c of item.content || []) {
                  const t = c.text || c.transcript;
                  if (t) texts.push(t);
                }
              }
              if (texts.length > 0) {
                return [...prev, { role: "assistant" as const, content: texts.join(""), timestamp: new Date() }];
              }
              return prev;
            });
          } else if (type === "error") {
            addMessage("status", `Error: ${data.error?.message || JSON.stringify(data.error)}`);
          }
        } catch {
          // ignore parse errors
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
    } catch (err: any) {
      addMessage("status", `Connection failed: ${err.message}`);
      setIsConnecting(false);
    }
  }, [accessToken, selectedModel, selectedVoice, customProxyBaseUrl, selectedGuardrails, addMessage, appendAssistantText, playAudioChunk]);

  const disconnect = useCallback(() => {
    stopRecording();
    wsRef.current?.close();
    wsRef.current = null;
    audioContextRef.current?.close();
    audioContextRef.current = null;
    nextPlayTimeRef.current = 0;
    configureSessionRef.current = false;
    setIsConnected(false);
  }, []);

  const startRecording = useCallback(async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    // Switch to server VAD mode for voice input
    wsRef.current.send(
      JSON.stringify({
        type: "session.update",
        session: {
          modalities: ["text", "audio"],
          voice: selectedVoice,
          input_audio_format: "pcm16",
          output_audio_format: "pcm16",
          input_audio_transcription: { model: "gpt-4o-mini-transcribe" },
          turn_detection: { type: "server_vad" },
        },
      })
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

        // Resample to 24kHz if needed
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

        // Convert to PCM16
        const pcm16 = new Int16Array(samples.length);
        for (let i = 0; i < samples.length; i++) {
          const s = Math.max(-1, Math.min(1, samples[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }

        // Base64 encode and send
        const bytes = new Uint8Array(pcm16.buffer);
        let binary = "";
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        const b64 = btoa(binary);

        wsRef.current!.send(
          JSON.stringify({ type: "input_audio_buffer.append", audio: b64 })
        );
      };

      source.connect(processor);
      processor.connect(ctx.destination);
      setIsRecording(true);
      addMessage("status", "🎙️ Listening...");
    } catch (err: any) {
      addMessage("status", `Microphone error: ${err.message}`);
    }
  }, [addMessage]);

  const stopRecording = useCallback(() => {
    processorRef.current?.disconnect();
    processorRef.current = null;
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    mediaStreamRef.current = null;
    setIsRecording(false);
  }, []);

  const configureSessionRef = useRef(false);

  const ensureTextSession = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    if (configureSessionRef.current) return;
    configureSessionRef.current = true;
    wsRef.current.send(
      JSON.stringify({
        type: "session.update",
        session: {
          modalities: ["text", "audio"],
          voice: selectedVoice,
          input_audio_format: "pcm16",
          output_audio_format: "pcm16",
          input_audio_transcription: { model: "gpt-4o-mini-transcribe" },
          turn_detection: null,
        },
      })
    );
  }, [selectedVoice]);

  const sendTextMessage = useCallback(() => {
    if (!inputText.trim() || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    const text = inputText.trim();
    addMessage("user", text);
    setInputText("");

    wsRef.current.send(
      JSON.stringify({
        type: "conversation.item.create",
        item: {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text }],
        },
      })
    );
    wsRef.current.send(JSON.stringify({ type: "response.create" }));
  }, [inputText, addMessage, ensureTextSession]);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
      audioContextRef.current?.close();
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center gap-3">
          <SoundOutlined className="text-lg text-blue-500" />
          <Text className="font-semibold text-gray-800">Realtime Voice Chat</Text>
          <span
            className={`inline-block w-2 h-2 rounded-full ${isConnected ? "bg-green-500" : "bg-gray-300"}`}
          />
          <Text className="text-xs text-gray-500">
            {isConnected ? "Connected" : isConnecting ? "Connecting..." : "Disconnected"}
          </Text>
        </div>
        <div className="flex items-center gap-2">
          <Select
            size="small"
            value={selectedVoice}
            onChange={setSelectedVoice}
            options={OPEN_AI_VOICE_SELECT_OPTIONS}
            style={{ width: 220 }}
            disabled={isConnected}
          />
          {!isConnected ? (
            <Button type="primary" onClick={connect} loading={isConnecting} size="small">
              Connect
            </Button>
          ) : (
            <Button danger onClick={disconnect} size="small" icon={<CloseCircleOutlined />}>
              Disconnect
            </Button>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 && !isConnected && (
          <div className="flex flex-col items-center justify-center h-full text-gray-400 gap-3">
            <SoundOutlined style={{ fontSize: 48 }} />
            <Text className="text-lg text-gray-500">Realtime Voice Playground</Text>
            <Text className="text-sm text-gray-400 text-center max-w-md">
              Click <b>Connect</b> to start a realtime session. You can speak using your microphone
              or type messages. The AI will respond with voice and text.
            </Text>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : msg.role === "status" ? "justify-center" : "justify-start"}`}
          >
            {msg.role === "status" ? (
              <div className="text-xs text-gray-400 italic px-3 py-1">{msg.content}</div>
            ) : (
              <div
                className={`max-w-[75%] rounded-2xl px-4 py-2.5 ${
                  msg.role === "user"
                    ? "bg-blue-500 text-white rounded-br-md"
                    : "bg-gray-100 text-gray-800 rounded-bl-md"
                }`}
              >
                <div className="text-xs font-medium mb-0.5 opacity-70">
                  {msg.role === "user" ? "You" : "AI"}
                </div>
                <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      {isConnected && (
        <div className="border-t border-gray-200 p-3 bg-white">
          <div className="flex items-center gap-2">
            <Button
              shape="circle"
              size="large"
              type={isRecording ? "primary" : "default"}
              danger={isRecording}
              icon={isRecording ? <AudioMutedOutlined /> : <AudioOutlined />}
              onClick={isRecording ? stopRecording : startRecording}
              title={isRecording ? "Stop recording" : "Start recording"}
              className={isRecording ? "animate-pulse" : ""}
            />
            <Input
              placeholder="Type a message or use the mic..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onPressEnter={sendTextMessage}
              className="flex-1"
              size="large"
            />
            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={sendTextMessage}
              disabled={!inputText.trim()}
              size="large"
            />
          </div>
          {isRecording && (
            <div className="mt-2 flex items-center gap-2 text-red-500 text-xs">
              <span className="inline-block w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              Listening — speak into your microphone. Server VAD will detect when you stop.
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default RealtimePlayground;
