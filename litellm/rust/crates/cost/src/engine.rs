//! The query engine: `price` hands each `PriceQuery` to its kind's handler, the way router-core runs one step per
//! module. A kind without a handler answers `NotPorted`, so its caller keeps pricing it the old way until it moves

use crate::query::PriceQuery;
use crate::receipt::{PriceOutcome, Unpriced};
use crate::row::RowSource;
use crate::{
    a2a, guardrail, image, mcp, ocr, rerank, search, speech, tokens, transcription, vector_store,
    video,
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
        PriceQuery::Session(_) | PriceQuery::BatchLine(_) => {
            Unpriced::NotPorted(query.kind()).into()
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::query::{Kind, Provider};
    use crate::testing::{TestSource, text_row, tokens_query, usage};

    #[rstest]
    fn a_kind_without_a_handler_is_not_ported(#[values(Kind::Session)] kind: Kind) {
        let header = tokens_query(Provider::Anthropic, "claude", usage(1, 1)).header;
        let query = PriceQuery::Session(header);
        let source = TestSource::with_row("claude", text_row(1.0, 1.0));
        assert_eq!(price(&source, &query), Unpriced::NotPorted(kind).into());
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
