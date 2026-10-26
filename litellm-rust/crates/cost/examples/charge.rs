use litellm_cost::money::UnitPrice;
use litellm_cost::price::{
    CacheReadMode, Channel, InputUnit, Meter, OutputUnit, Price, Prices, ServiceTier, TokenKind,
};
use litellm_cost::{Priced, Usage, When, calculate};
use time::OffsetDateTime;

fn main() {
    let input = Meter::Input(InputUnit::Token(TokenKind::Text));
    let output = Meter::Output(OutputUnit::Token(TokenKind::Text));
    let cache = Meter::CacheRead {
        kind: TokenKind::Text,
        mode: CacheReadMode::Implicit,
    };
    let prices = Prices::new(
        [
            (input.clone(), 2.0),
            (output.clone(), 4.0),
            (cache.clone(), 0.5),
        ]
        .into_iter()
        .map(|(meter, rate)| {
            (
                Price::online(meter, Default::default()),
                UnitPrice::new(rate).unwrap(),
            )
        })
        .collect(),
        None,
    );
    let usage = Usage {
        quantities: [(input, 75), (cache, 25), (output, 20)]
            .map(|(meter, count)| (meter, litellm_cost::Quantity::Count(count)))
            .into(),
        prompt_tokens: 100,
        ..Usage::default()
    };
    let when = When {
        tier: ServiceTier::Standard,
        channel: Channel::Online,
        at: OffsetDateTime::UNIX_EPOCH,
    };
    match calculate(&prices, &usage, &when) {
        Priced::Known(cost) => println!(
            "input={} output={} total={}",
            cost.input().get(),
            cost.output().get(),
            cost.total.get()
        ),
        Priced::Unpriced { missing, partial } => {
            println!("unpriced={missing:?} partial={}", partial.total.get())
        }
    }
}
