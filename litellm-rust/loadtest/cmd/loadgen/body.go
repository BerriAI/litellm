package main

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"strconv"
)

// bytesPerToken approximates English text under common tokenizers; only the byte count matters to
// the proxy, so this sets body size, not exact token counts.
const bytesPerToken = 4

// turnTokens is the size of each conversation turn; a large context becomes many messages, the way
// long agent sessions look, rather than one giant string.
const turnTokens = 8000

var vocabulary = []string{
	"the", "of", "and", "to", "in", "is", "that", "for", "it", "as", "was", "with", "be", "by", "on", "not",
	"this", "are", "or", "from", "at", "which", "but", "have", "an", "they", "you", "were", "there", "been",
	"one", "all", "would", "their", "will", "more", "when", "who", "what", "about", "into", "than", "them",
	"can", "only", "other", "new", "some", "could", "time", "these", "two", "may", "then", "first", "any",
	"request", "latency", "gateway", "proxy", "token", "stream", "provider", "router", "deployment", "budget",
	"cache", "metric", "handler", "response", "payload", "context", "window", "function", "result", "value",
}

// textPool is a block of lowercase words and spaces, so it can be copied into JSON strings without
// escaping. Bodies are slices of it at random offsets.
type textPool []byte

func newTextPool(size int, seed uint64) textPool {
	r := rand.New(rand.NewPCG(seed, 7))
	var b bytes.Buffer
	b.Grow(size + 32)
	for b.Len() < size {
		b.WriteString(vocabulary[r.IntN(len(vocabulary))])
		b.WriteByte(' ')
	}
	return textPool(b.Bytes()[:size])
}

func (p textPool) slice(r *rand.Rand, n int) []byte {
	n = min(n, len(p))
	off := r.IntN(len(p) - n + 1)
	return p[off : off+n]
}

func buildBody(pool textPool, r *rand.Rand, res result, outTokens int, nonce string) []byte {
	total := res.CtxTokens * bytesPerToken
	var b bytes.Buffer
	b.Grow(total + 1024)
	b.WriteString(`{"model":`)
	b.WriteString(strconv.Quote(res.Model))
	b.WriteString(`,"max_tokens":`)
	b.WriteString(strconv.Itoa(outTokens))
	if res.Stream {
		b.WriteString(`,"stream":true`)
		if res.Endpoint == "chat" {
			b.WriteString(`,"stream_options":{"include_usage":true}`)
		}
	} else {
		b.WriteString(`,"stream":false`)
	}
	b.WriteString(`,"messages":[`)
	remaining := total
	role := "user"
	first := true
	for {
		n := min(remaining, turnTokens*bytesPerToken)
		if !first {
			b.WriteByte(',')
		}
		b.WriteString(`{"role":"` + role + `","content":"`)
		if first {
			b.WriteString("conversation " + nonce + " ")
			first = false
		}
		b.Write(pool.slice(r, n))
		remaining -= n
		if remaining <= 0 && role == "user" {
			// The directive goes last so mockllm finds it by scanning only the tail.
			fmt.Fprintf(&b, " [[mock ttft_ms=%d dur_ms=%d out=%d]]", res.TTFTPlanMs, res.DurPlanMs, outTokens)
			b.WriteString(`"}`)
			break
		}
		b.WriteString(`"}`)
		if role == "user" {
			role = "assistant"
		} else {
			role = "user"
		}
	}
	b.WriteString("]}")
	return b.Bytes()
}
