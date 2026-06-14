mod backend;
mod cli;
mod model;
mod policy;
mod probe;
mod store;

use anyhow::Result;
use clap::Parser;

use crate::cli::{Cli, run};

fn main() -> Result<()> {
    run(Cli::parse())
}
