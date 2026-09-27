use litellm_cost::error::CostError;

use litellm_cost::Rate;
use litellm_cost::non_token::{
    Charge, Component, ImageRates, ImageUsage, OcrRates, OcrUsage, Unit, calculate,
    calculate_image, calculate_ocr,
};
use rstest::rstest;

#[rstest]
fn combines_non_token_charges_without_losing_components() {
    let charges = [
        Charge {
            unit: Unit::Image,
            quantity: 2.0,
            rate: 0.04,
            units_per_rate: 1.0,
        },
        Charge {
            unit: Unit::Second,
            quantity: 2.5,
            rate: 0.02,
            units_per_rate: 1.0,
        },
        Charge {
            unit: Unit::Page,
            quantity: 3.0,
            rate: 0.01,
            units_per_rate: 1.0,
        },
    ];

    let result = calculate(&charges).unwrap();
    assert_eq!(
        result.components,
        vec![
            Component {
                unit: Unit::Image,
                amount: 0.08,
            },
            Component {
                unit: Unit::Second,
                amount: 0.05,
            },
            Component {
                unit: Unit::Page,
                amount: 0.03,
            },
        ]
    );
    assert!((result.total - 0.16).abs() < 1e-12);
}

#[rstest]
fn supports_prices_per_thousand_units_and_free_usage() {
    let charges = [
        Charge {
            unit: Unit::GuardrailUnit,
            quantity: 2500.0,
            rate: 0.6,
            units_per_rate: 1000.0,
        },
        Charge {
            unit: Unit::Request,
            quantity: 1.0,
            rate: 0.0,
            units_per_rate: 1.0,
        },
    ];

    let result = calculate(&charges).unwrap();
    assert_eq!(result.components[0].amount, 1.5);
    assert_eq!(result.components[1].amount, 0.0);
    assert_eq!(result.total, 1.5);
}

#[rstest]
fn rejects_invalid_inputs_and_overflow() {
    let valid = Charge {
        unit: Unit::Credit,
        quantity: 1.0,
        rate: 0.5,
        units_per_rate: 1.0,
    };

    assert_eq!(
        calculate(&[Charge {
            quantity: -1.0,
            ..valid
        }]),
        Err(CostError::InvalidQuantity)
    );
    assert_eq!(
        calculate(&[Charge {
            rate: f64::NAN,
            ..valid
        }]),
        Err(CostError::InvalidRate)
    );
    assert_eq!(
        calculate(&[Charge {
            units_per_rate: 0.0,
            ..valid
        }]),
        Err(CostError::InvalidRateBasis)
    );
    assert_eq!(
        calculate(&[Charge {
            quantity: f64::MAX,
            rate: f64::MAX,
            ..valid
        }]),
        Err(CostError::NonFiniteCost)
    );
}

#[rstest]
fn image_price_uses_first_priced_table_and_preserves_zero() {
    let deployment = ImageRates {
        input_per_image: Rate::Missing,
        output_per_image: Rate::Null,
        input_per_pixel: Rate::Value(0.001),
    };
    let published = ImageRates {
        input_per_image: Rate::Value(0.5),
        output_per_image: Rate::Missing,
        input_per_pixel: Rate::Missing,
    };
    let usage = ImageUsage {
        count: 2,
        width: 10,
        height: 20,
    };

    let result = calculate_image(&[deployment, published], usage).unwrap();
    assert_eq!(result.components[0].unit, Unit::Pixel);
    assert_eq!(result.total, 0.4);

    let free = ImageRates {
        input_per_image: Rate::Value(0.0),
        ..deployment
    };
    assert_eq!(
        calculate_image(&[free, published], usage).unwrap().total,
        0.0
    );
}

#[rstest]
fn ocr_credits_take_precedence_and_annotation_uses_page_fallback() {
    let rates = OcrRates {
        per_credit: Rate::Value(0.2),
        per_page: Rate::Value(0.03),
        per_annotation_page: Rate::Missing,
    };
    let usage = OcrUsage {
        credits: Some(4.0),
        pages: 3,
        annotation_pages: 2,
    };

    let credit_cost = calculate_ocr(rates, usage).unwrap();
    assert_eq!(credit_cost.components[0].unit, Unit::Credit);
    assert_eq!(credit_cost.total, 0.8);

    let page_cost = calculate_ocr(
        OcrRates {
            per_credit: Rate::Missing,
            ..rates
        },
        usage,
    )
    .unwrap();
    assert_eq!(page_cost.components.len(), 2);
    assert!((page_cost.total - 0.15).abs() < 1e-12);
}

#[rstest]
fn image_rejects_missing_prices_and_ocr_returns_zero() {
    let image = ImageUsage {
        count: 1,
        width: 10,
        height: 10,
    };
    assert_eq!(calculate_image(&[], image), Err(CostError::MissingRate));
    assert_eq!(
        calculate_ocr(
            OcrRates {
                per_credit: Rate::Missing,
                per_page: Rate::Null,
                per_annotation_page: Rate::Missing,
            },
            OcrUsage {
                credits: None,
                pages: 1,
                annotation_pages: 0,
            },
        ),
        Ok(litellm_cost::non_token::Cost {
            components: vec![],
            total: 0.0,
        })
    );
}
