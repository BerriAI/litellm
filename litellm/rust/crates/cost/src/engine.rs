//! The query engine: `price` hands each `PriceQuery` to its kind's handler, the way router-core runs one step per
//! module. A case a handler does not price yet answers `NotPorted`, so its caller keeps pricing it the old way

use crate::query::PriceQuery;
use crate::receipt::PriceOutcome;
use crate::row::RowSource;
use crate::{
    a2a, batch, guardrail, image, mcp, ocr, rerank, search, session, speech, tokens, tools,
    transcription, vector_store, video,
};

pub fn price(source: &impl RowSource, query: &PriceQuery) -> PriceOutcome {
    match query {
        PriceQuery::Tokens(tokens) => tokens::price(source, tokens),
        PriceQuery::Ocr(ocr) => ocr::price(source, ocr),
        PriceQuery::Rerank(rerank) => rerank::price(source, rerank),
        PriceQuery::VectorStoreSearch(search) => vector_store::price(source, search),
        PriceQuery::Speech(speech) => speech::price(source, speech),
        PriceQuery::Transcription(transcription) => transcription::price(source, transcription),
        PriceQuery::Search(search) => search::price(source, search),
        PriceQuery::Video(video) => video::price(source, video),
        PriceQuery::Image(image) => image::price(source, image),
        PriceQuery::Guardrail(guardrail) => guardrail::price(source, guardrail),
        PriceQuery::A2a(a2a) => a2a::price(source, a2a),
        PriceQuery::Mcp(mcp) => mcp::price(source, mcp),
        PriceQuery::BatchLine(line) => batch::price(source, line),
        PriceQuery::Session(session) => session::price(source, session),
        PriceQuery::Tools(tools) => tools::price(source, tools),
    }
}

/// Prices each query against the same rows, in order, so a caller with many calls, such as the lines of a batch
/// output file, hands them over at once and every row is read once
pub fn price_many<'q, S: RowSource>(
    source: &S,
    queries: impl IntoIterator<Item = &'q PriceQuery>,
) -> Vec<PriceOutcome> {
    queries
        .into_iter()
        .map(|query| price(source, query))
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::query::Provider;
    use crate::testing::{TestSource, text_row, tokens_query, usage};

    #[rstest]
    fn many_queries_are_priced_in_order_each_as_alone() {
        let source = TestSource::with_row("claude", text_row(1.0, 2.0));
        let queries = [usage(3, 4), usage(1, 0)].map(|used| {
            PriceQuery::Tokens(Box::new(tokens_query(Provider::Anthropic, "claude", used)))
        });
        let missing = PriceQuery::Tokens(Box::new(tokens_query(
            Provider::Anthropic,
            "unknown",
            usage(1, 1),
        )));
        let all: Vec<&PriceQuery> = queries.iter().chain([&missing]).collect();
        assert_eq!(
            price_many(&source, all.iter().copied()),
            all.iter()
                .map(|query| price(&source, query))
                .collect::<Vec<_>>()
        );
    }

    #[rstest]
    fn the_tokens_kind_is_priced() {
        let source = TestSource::with_row("claude", text_row(1.0, 2.0));
        let query = PriceQuery::Tokens(Box::new(tokens_query(
            Provider::Anthropic,
            "claude",
            usage(3, 4),
        )));
        let PriceOutcome::Priced(receipt) = price(&source, &query) else {
            panic!("a priced row prices");
        };
        assert_eq!(receipt.total, 11.0);
    }
}
