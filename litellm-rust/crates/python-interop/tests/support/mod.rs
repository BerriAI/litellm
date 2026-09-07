mod callback_owner;
pub mod python;
pub mod scenarios;

#[derive(Clone, Copy, Debug)]
pub enum Backend {
    Python,
    PreparedCall,
}
