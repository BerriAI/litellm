pub trait StreamTransformer {
    type Input;
    type Output;
    type Error;

    fn transform(&mut self, input: Self::Input) -> Result<Vec<Self::Output>, Self::Error>;

    fn finish(&mut self) -> Result<Vec<Self::Output>, Self::Error>;
}
