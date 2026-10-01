use litellm_accounting::{
    Charges, Cost, Error, Outcome, ProviderWork, ReportedUsage, Terminal, Usd,
};
use rstest::{fixture, rstest};
use rust_decimal::Decimal;
use rusty_money::{Money, MoneyError, iso};

fn usd(value: &str) -> Usd {
    Money::from_str(value, iso::USD).unwrap()
}

#[fixture]
fn services() -> Cost {
    Cost::Known(usd("2"))
}

#[rstest]
#[case::provider(Cost::Known(usd("-1")), Cost::Known(usd("0")))]
#[case::services(Cost::Known(usd("0")), Cost::Known(usd("-1")))]
fn negative_charges_cannot_enter_accounting(#[case] provider: Cost, #[case] services: Cost) {
    assert_eq!(Charges::new(provider, services), Err(Error::NegativeCost));
}

#[rstest]
#[case::provider(Cost::Known(Money::from_major(1, iso::EUR)), Cost::Known(usd("0")))]
#[case::services(Cost::Known(usd("0")), Cost::Known(Money::from_major(1, iso::EUR)))]
fn non_usd_charges_cannot_enter_accounting(#[case] provider: Cost, #[case] services: Cost) {
    assert_eq!(
        Charges::new(provider, services),
        Err(Error::InvalidCurrency {
            currency: iso::EUR.iso_alpha_code,
        })
    );
}

#[rstest]
fn overflowing_totals_cannot_be_selected_for_settlement() {
    let maximum = Cost::Known(Money::from_decimal(Decimal::MAX, iso::USD));
    let charges = Charges::new(maximum, maximum).unwrap();
    assert_eq!(charges.total(), Err(Error::Money(MoneyError::Overflow)));
    assert_eq!(
        Terminal::<()>::new(
            Outcome::Succeeded,
            ProviderWork::Started,
            ReportedUsage::Unknown,
            charges,
        ),
        Err(Error::Money(MoneyError::Overflow))
    );
}

#[rstest]
#[case::unknown_provider(Cost::Unknown, Cost::Known(usd("0")))]
#[case::unknown_services(Cost::Known(usd("0")), Cost::Unknown)]
#[case::both_unknown(Cost::Unknown, Cost::Unknown)]
fn unknown_cost_is_never_a_known_zero(#[case] provider: Cost, #[case] services: Cost) {
    let charges = Charges::new(provider, services).unwrap();
    assert_eq!(charges.total(), Ok(Cost::Unknown));
    assert_eq!(charges.provider(), provider);
    assert_eq!(charges.services(), services);
}

#[rstest]
#[case::decimal("0.1", "0.2", "0.3")]
#[case::fractional_cent("0.000000001", "0.000000002", "0.000000003")]
#[case::zero("0", "0", "0")]
fn decimal_totals_preserve_amounts_and_breakdown(
    #[case] provider_amount: &str,
    #[case] service_amount: &str,
    #[case] expected_amount: &str,
) {
    let provider = Cost::Known(usd(provider_amount));
    let services = Cost::Known(usd(service_amount));
    let charges = Charges::new(provider, services).unwrap();
    assert_eq!(charges.total(), Ok(Cost::Known(usd(expected_amount))));
    assert_eq!(charges.provider(), provider);
    assert_eq!(charges.services(), services);
}

#[rstest]
#[case::success(Outcome::Succeeded)]
#[case::failure(Outcome::Failed)]
#[case::cancellation(Outcome::Cancelled)]
fn avoided_provider_work_preserves_reported_usage_and_service_charges(
    services: Cost,
    #[case] outcome: Outcome,
) {
    let usage = ReportedUsage::Known((100_u64, 20_u64));
    let terminal = Terminal::new(
        outcome,
        ProviderWork::NotStarted,
        usage.clone(),
        Charges::new(Cost::Known(usd("3")), services).unwrap(),
    )
    .unwrap();
    assert_eq!(terminal.usage(), &usage);
    assert_eq!(terminal.work(), ProviderWork::NotStarted);
    assert_eq!(terminal.charges().provider(), Cost::Known(usd("0")));
    assert_eq!(terminal.charges().services(), services);
    assert_eq!(terminal.charges().total(), Ok(services));
    assert_eq!(terminal.outcome(), outcome);
}

#[rstest]
fn avoided_provider_work_does_not_make_unknown_service_cost_free() {
    let terminal = Terminal::<()>::new(
        Outcome::Succeeded,
        ProviderWork::NotStarted,
        ReportedUsage::Unknown,
        Charges::new(Cost::Unknown, Cost::Unknown).unwrap(),
    )
    .unwrap();
    assert_eq!(terminal.charges().provider(), Cost::Known(usd("0")));
    assert_eq!(terminal.charges().services(), Cost::Unknown);
    assert_eq!(terminal.charges().total(), Ok(Cost::Unknown));
    assert_eq!(terminal.usage(), &ReportedUsage::Unknown);
}

#[rstest]
#[case::success(Outcome::Succeeded)]
#[case::failure(Outcome::Failed)]
#[case::cancellation(Outcome::Cancelled)]
fn terminal_outcomes_do_not_erase_incurred_charges(services: Cost, #[case] outcome: Outcome) {
    let charges = Charges::new(Cost::Known(usd("3")), services).unwrap();
    let terminal = Terminal::new(
        outcome,
        ProviderWork::Started,
        ReportedUsage::Known((10_u64, 1_u64)),
        charges,
    )
    .unwrap();
    assert_eq!(terminal.outcome(), outcome);
    assert_eq!(terminal.charges(), charges);
    assert_eq!(terminal.usage(), &ReportedUsage::Known((10, 1)));
}

#[rstest]
#[case::known(costs_for_unknown_work())]
#[case::unknown(Charges::new(Cost::Unknown, Cost::Known(usd("1"))).unwrap())]
fn unknown_provider_work_does_not_imply_free_generation(#[case] charges: Charges) {
    let terminal = Terminal::<()>::new(
        Outcome::Failed,
        ProviderWork::Unknown,
        ReportedUsage::Unknown,
        charges,
    )
    .unwrap();
    assert_eq!(terminal.charges(), charges);
}

fn costs_for_unknown_work() -> Charges {
    Charges::new(Cost::Known(usd("3")), Cost::Known(usd("1"))).unwrap()
}
