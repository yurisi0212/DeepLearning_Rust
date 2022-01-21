use std::fs::File;
use std::path::Path;

use polars::prelude::*;
use polars::frame::DataFrame;
use polars::prelude::Result as PolarResult;
use polars::prelude::SerReader;

pub fn read_csv_with_schema<P: AsRef<Path>+ std::fmt::Debug>(path: P) -> PolarResult<DataFrame> {
    let schema = Schema::new(vec![
        Field::new("species", DataType::Utf8),
        Field::new("island", DataType::Utf8),
        Field::new("culmen_length_mm", DataType::Float64),
        Field::new("culmen_depth_mm", DataType::Float64),
        Field::new("flipper_length_mm", DataType::Float64),
        Field::new("body_mass_g", DataType::Float64),
        Field::new("sex", DataType::Utf8),
    ]);
    
    println!("{:?}", path);

    let file = File::open(path).expect("Cannot open file.");
    CsvReader::new(file)
        .with_schema(Arc::new(schema))
        .has_header(true)
        .with_ignore_parser_errors(true)
        .finish()
}