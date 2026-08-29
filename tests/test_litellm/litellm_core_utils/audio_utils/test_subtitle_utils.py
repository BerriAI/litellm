from litellm.litellm_core_utils.audio_utils.subtitle_utils import (
    SubtitleToken,
    _merge_tokens_into_words,
    render_subtitle_tokens_as_srt,
    render_subtitle_tokens_as_vtt,
    synthesize_subtitle_document,
)


class TestRenderSubtitleTokensAsSrt:
    def test_single_cue_full_document(self):
        tokens = (
            SubtitleToken(text="Hello ", start_ms=0, end_ms=500),
            SubtitleToken(text="world.", start_ms=500, end_ms=1000),
        )
        assert render_subtitle_tokens_as_srt(tokens) == "1\n00:00:00,000 --> 00:00:01,000\nHello world.\n"

    def test_speaker_change_starts_a_new_cue(self):
        tokens = (
            SubtitleToken(text="Hi.", start_ms=0, end_ms=1000, speaker="spk:0"),
            SubtitleToken(text="Hey.", start_ms=1500, end_ms=2500, speaker="spk:1"),
        )
        assert render_subtitle_tokens_as_srt(tokens) == (
            "1\n00:00:00,000 --> 00:00:01,000\nHi.\n\n2\n00:00:01,500 --> 00:00:02,500\nHey.\n"
        )

    def test_width_budget_starts_a_new_cue_at_word_boundaries(self):
        tokens = tuple(SubtitleToken(text="abcdefghi ", start_ms=i * 100, end_ms=i * 100 + 90) for i in range(20))
        result = render_subtitle_tokens_as_srt(tokens)
        texts = [cue.split("\n", 2)[2] for cue in result.strip().split("\n\n")]
        assert len(texts) == 3
        assert all(len(text) <= 84 for text in texts)
        assert all(set(text.split()) == {"abcdefghi"} for text in texts)

    def test_duration_cap_starts_a_new_cue_before_word_crossing_7000ms(self):
        tokens = (
            SubtitleToken(text="Alpha ", start_ms=0, end_ms=3400),
            SubtitleToken(text="beta ", start_ms=3400, end_ms=6800),
            SubtitleToken(text="gamma", start_ms=6800, end_ms=7400),
        )
        assert render_subtitle_tokens_as_srt(tokens) == (
            "1\n00:00:00,000 --> 00:00:06,800\nAlpha beta\n\n2\n00:00:06,800 --> 00:00:07,400\ngamma\n"
        )

    def test_silence_gap_starts_a_new_cue(self):
        tokens = (
            SubtitleToken(text="Alpha ", start_ms=0, end_ms=400),
            SubtitleToken(text="beta", start_ms=2000, end_ms=2400),
        )
        assert render_subtitle_tokens_as_srt(tokens) == (
            "1\n00:00:00,000 --> 00:00:00,400\nAlpha\n\n2\n00:00:02,000 --> 00:00:02,400\nbeta\n"
        )

    def test_sentence_final_punctuation_starts_a_new_cue(self):
        tokens = (
            SubtitleToken(text="Done. ", start_ms=0, end_ms=400),
            SubtitleToken(text="Next", start_ms=500, end_ms=800),
        )
        assert render_subtitle_tokens_as_srt(tokens) == (
            "1\n00:00:00,000 --> 00:00:00,400\nDone.\n\n2\n00:00:00,500 --> 00:00:00,800\nNext\n"
        )

    def test_subword_tokens_merge_into_words_before_grouping(self):
        tokens = (
            SubtitleToken(text=" hel", start_ms=0, end_ms=150),
            SubtitleToken(text="lo", start_ms=150, end_ms=300),
            SubtitleToken(text=" world.", start_ms=350, end_ms=600),
        )
        assert render_subtitle_tokens_as_srt(tokens) == "1\n00:00:00,000 --> 00:00:00,600\nhello world.\n"

    def test_cjk_tokens_merge_and_keep_punctuation_attached(self):
        tokens = (
            SubtitleToken(text="編", start_ms=0, end_ms=100),
            SubtitleToken(text="集", start_ms=100, end_ms=200),
            SubtitleToken(text="、", start_ms=200, end_ms=250),
            SubtitleToken(text="保存", start_ms=250, end_ms=400),
        )
        assert [word.text for word in _merge_tokens_into_words(tokens)] == ["編", "集、", "保存"]

    def test_timestampless_token_joins_the_current_cue(self):
        tokens = (
            SubtitleToken(text="Hello ", start_ms=0, end_ms=500),
            SubtitleToken(text="there "),
            SubtitleToken(text="world.", start_ms=900, end_ms=1300),
        )
        assert render_subtitle_tokens_as_srt(tokens) == "1\n00:00:00,000 --> 00:00:01,300\nHello there world.\n"

    def test_only_timestampless_tokens_renders_empty(self):
        assert render_subtitle_tokens_as_srt((SubtitleToken(text="no timestamps"),)) == ""

    def test_empty_tokens_render_empty(self):
        assert render_subtitle_tokens_as_srt(()) == ""

    def test_timestamps_past_one_hour(self):
        tokens = (SubtitleToken(text="Late.", start_ms=3_661_001, end_ms=3_662_002),)
        assert render_subtitle_tokens_as_srt(tokens) == "1\n01:01:01,001 --> 01:01:02,002\nLate.\n"

    def test_negative_timestamps_clamp_to_zero(self):
        tokens = (SubtitleToken(text="Early.", start_ms=-100, end_ms=-50),)
        assert render_subtitle_tokens_as_srt(tokens) == "1\n00:00:00,000 --> 00:00:00,000\nEarly.\n"

    def test_missing_end_falls_back_to_cue_start(self):
        tokens = (SubtitleToken(text="Open.", start_ms=1200),)
        assert render_subtitle_tokens_as_srt(tokens) == "1\n00:00:01,200 --> 00:00:01,200\nOpen.\n"


class TestRenderSubtitleTokensAsVtt:
    def test_single_cue_full_document(self):
        tokens = (
            SubtitleToken(text="Hello ", start_ms=0, end_ms=500),
            SubtitleToken(text="world.", start_ms=500, end_ms=1000),
        )
        assert render_subtitle_tokens_as_vtt(tokens) == "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello world.\n"

    def test_empty_tokens_render_header_only(self):
        assert render_subtitle_tokens_as_vtt(()) == "WEBVTT\n"

    def test_timestamps_past_one_hour_use_dot_separator(self):
        tokens = (SubtitleToken(text="Late.", start_ms=3_661_001, end_ms=3_662_002),)
        assert render_subtitle_tokens_as_vtt(tokens) == "WEBVTT\n\n01:01:01.001 --> 01:01:02.002\nLate.\n"

    def test_speaker_change_starts_a_new_cue(self):
        tokens = (
            SubtitleToken(text="Hi.", start_ms=0, end_ms=1000, speaker=1),
            SubtitleToken(text="Hey.", start_ms=1500, end_ms=2500, speaker=2),
        )
        assert render_subtitle_tokens_as_vtt(tokens) == (
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHi.\n\n00:00:01.500 --> 00:00:02.500\nHey.\n"
        )


class TestSynthesizeSubtitleDocument:
    WORDS = [
        {"word": "Four", "start": 0.4, "end": 0.7, "speaker": "spk:0"},
        {"word": "score", "start": 0.7, "end": 1.1, "speaker": "spk:0"},
    ]

    def test_srt_from_words_converts_seconds_to_milliseconds(self):
        assert synthesize_subtitle_document(self.WORDS, "srt") == "1\n00:00:00,400 --> 00:00:01,100\nFour score\n"

    def test_vtt_from_words_converts_seconds_to_milliseconds(self):
        assert synthesize_subtitle_document(self.WORDS, "vtt") == (
            "WEBVTT\n\n00:00:00.400 --> 00:00:01.100\nFour score\n"
        )

    def test_speaker_change_splits_cues(self):
        words = [
            {"word": "Hi", "start": 0.0, "end": 0.5, "speaker": "spk:0"},
            {"word": "Hey", "start": 0.6, "end": 1.0, "speaker": "spk:1"},
        ]
        assert synthesize_subtitle_document(words, "srt") == (
            "1\n00:00:00,000 --> 00:00:00,500\nHi\n\n2\n00:00:00,600 --> 00:00:01,000\nHey\n"
        )

    def test_non_subtitle_format_returns_none(self):
        assert synthesize_subtitle_document(self.WORDS, "verbose_json") is None
        assert synthesize_subtitle_document(self.WORDS, "json") is None

    def test_missing_words_returns_none(self):
        assert synthesize_subtitle_document(None, "srt") is None
        assert synthesize_subtitle_document([], "srt") is None

    def test_words_without_timestamps_return_none(self):
        assert synthesize_subtitle_document([{"word": "Hello"}], "srt") is None
        assert synthesize_subtitle_document([{"word": "Hello"}], "vtt") is None

    def test_malformed_words_return_none(self):
        assert synthesize_subtitle_document("not words", "srt") is None
        assert synthesize_subtitle_document([{"word": "ok", "start": "not-a-number"}], "srt") is None
