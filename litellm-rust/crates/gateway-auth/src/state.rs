pub trait MasterKeyState {
    fn master_key(&self) -> Option<&str>;
}
