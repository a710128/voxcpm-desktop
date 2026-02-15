use std::path::PathBuf;

use voxcpm_engine_sdk::EngineSdk;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .expect("usage: list_devices <engine_bin>");

    let sdk = EngineSdk::spawn(engine).await?;
    let devices = sdk.list_devices(1).await?;
    println!("{devices:?}");
    Ok(())
}
