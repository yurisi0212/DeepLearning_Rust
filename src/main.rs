use std::env;
use std::path::Path;

use deep_learning_rust::utils::csv;

use polars::prelude::*;
use polars::frame::DataFrame;
use polars::prelude::Result as PolarResult;

use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::BaseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::model_selection::train_test_split;

use smartcore::metrics::mean_squared_error;
use smartcore::metrics::accuracy;

fn get_feature_target(df: &DataFrame) -> (PolarResult<DataFrame>, PolarResult<DataFrame>) {
    let features = df.select(vec![
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]);
    let target = df.select("species");
    (features, target)
}

pub fn convert_features_to_matrix(df: &DataFrame) -> Result<DenseMatrix<f64>> {
    let nrows = df.height();
    let ncols = df.width();

    let features_res = df.to_ndarray::<Float64Type>().unwrap();
    let mut xmatrix: DenseMatrix<f64> = BaseMatrix::zeros(nrows, ncols);
    let mut col: u32 = 0;
    let mut row: u32 = 0;

    for val in features_res.iter() {
        let m_row = usize::try_from(row).unwrap();
        let m_col = usize::try_from(col).unwrap();
        xmatrix.set(m_row, m_col, *val);
        // check what we have to update
        if m_col == ncols - 1 {
            row += 1;
            col = 0;
        } else {
            col += 1;
        }
    }

    Ok(xmatrix)
}

fn str_to_num(str_val: &Series) -> Series {
    str_val
        .utf8()
        .unwrap()
        .into_iter()
        .map(|opt_name: Option<&str>| {
            opt_name.map(|name: &str| {
                match name {
                    "Adelie" => 1,
                    "Chinstrap" => 2,
                    "Gentoo" => 3,
                    _ => panic!("Problem species str to num"),
                }
            })
        })
        .collect::<UInt32Chunked>()
        .into_series()
}

fn main(){
    let file = Path::new("src/data/penguins_size.csv");
    let path = env::current_dir().unwrap();
    let csv = path.join(file);
    let df: DataFrame = csv::read_csv_with_schema(&csv).unwrap();
    let df2 = df.drop_nulls(None).unwrap();

    let (feature, target) = get_feature_target(&df2);

    let xmatrix = convert_features_to_matrix(&feature.unwrap());

    let target_array = target
        .unwrap()
        .apply("species", str_to_num)
        .unwrap()
        .to_ndarray::<Float64Type>()
        .unwrap();

    let mut y: Vec<f64> = Vec::new();
    for val in target_array.iter() {
        y.push(*val);
    }

    let (x_train, x_test, y_train, y_test) = train_test_split(&xmatrix.unwrap(), &y, 0.3, true);

    let reg = LogisticRegression::fit(&x_train, &y_train, Default::default()).unwrap();

    let preds = reg.predict(&x_test).unwrap();
    let mse = mean_squared_error(&y_test, &preds);
    println!("MSE: {}", mse);
    println!("accuracy : {}", accuracy(&y_test, &preds));
}