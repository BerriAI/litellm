#![forbid(unsafe_code)]

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("bench") => println!("would sweep the matrix"),
        _ => println!("usage: cargo xtask bench"),
    }
}
