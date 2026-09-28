use rusty_money::{Money, iso};

use crate::Error;

pub type Usd = Money<'static, iso::Currency>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cost {
    Unknown,
    Known(Usd),
}

impl Cost {
    fn validate(self) -> Result<(), Error> {
        let Self::Known(money) = self else {
            return Ok(());
        };
        if money.currency() != iso::USD {
            return Err(Error::InvalidCurrency {
                currency: money.currency().iso_alpha_code,
            });
        }
        if money.is_negative() {
            return Err(Error::NegativeCost);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProviderWork {
    NotStarted,
    Started,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ReportedUsage<U> {
    Unknown,
    Known(U),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Charges {
    provider: Cost,
    services: Cost,
}

impl Charges {
    pub fn new(provider: Cost, services: Cost) -> Result<Self, Error> {
        provider.validate()?;
        services.validate()?;
        Ok(Self { provider, services })
    }

    pub fn provider(self) -> Cost {
        self.provider
    }

    pub fn services(self) -> Cost {
        self.services
    }

    pub fn total(self) -> Result<Cost, Error> {
        match (self.provider, self.services) {
            (Cost::Known(provider), Cost::Known(services)) => {
                provider.add(services).map(Cost::Known).map_err(Error::from)
            }
            _ => Ok(Cost::Unknown),
        }
    }

    pub(crate) fn for_work(self, work: ProviderWork) -> Self {
        match work {
            ProviderWork::Started => self,
            ProviderWork::NotStarted => Self {
                provider: Cost::Known(Money::from_major(0, iso::USD)),
                services: self.services,
            },
        }
    }
}
