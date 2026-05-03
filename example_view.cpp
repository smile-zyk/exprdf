#include <QApplication>
#include <exprdf/exprdf.hpp>
#include <exprdf/UnitFormat.hpp>
#include <exprdf/qt/DataFrameView.h>
#include <cmath>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    // DataFrame matching the screenshot:
    // level (uniform outer): 36.654, 41.654
    // number (grouped inner): each level has its own list of numbers (all =1 here, but different lengths)
    // real_indexs11, PAE_contours: data columns

    auto df = std::make_shared<exprdf::DataFrame>();
    df->set_name("PAE Contours");

    // --- level: uniform outer index (2 levels) ---
    df->add_uniform_index<double>("level", {36.654, 41.654});

    // --- number: grouped inner index (ragged: 32 entries for level0, 24 entries for level1) ---
    df->add_uniform_index<int>("number", {1});

    // --- real_indexs11 data (level 36.654: row 0..31, level 41.654: row 32..55) ---
    std::vector<std::vector<double>> real_s11 = {
        // level = 36.654 (32 rows)
        { 0.134,  0.102,  0.080,  0.032,  0.017, -0.053, -0.057, -0.081,
        -0.146, -0.226, -0.262, -0.292, -0.298, -0.306, -0.301, -0.275,
        -0.271, -0.264, -0.208, -0.151, -0.144, -0.141, -0.089, -0.043,
        -0.030, -0.003,  0.018,  0.027,  0.061,  0.080,  0.100,  0.122
        },
        // level = 41.654 (24 rows)
        { 0.254,  0.238,  0.171,  0.108,  0.090,  0.066,  0.014, -0.001,
        -0.004, -0.029, -0.022, -0.004, -0.002, -0.001,  0.031,  0.060,
         0.073,  0.086,  0.120,  0.162,  0.174,  0.207,  0.221,  0.224
        }
    };

    // --- PAE_contours data ---
    std::vector<double> pae = {
        // level = 36.654 (32 rows)
         0.586,  0.574,  0.567,  0.551,  0.544,  0.519,  0.516,  0.505,
         0.448,  0.370,  0.305,  0.235,  0.201,  0.100,  0.064, -0.035,
        -0.043, -0.055, -0.115, -0.170, -0.175, -0.177, -0.240, -0.263,
        -0.285, -0.305, -0.329, -0.331, -0.361, -0.367, -0.385, -0.395,
        // level = 41.654 (24 rows)
         0.489,  0.478,  0.455,  0.400,  0.391,  0.370,  0.260,  0.235,
         0.212,  0.100,  0.078, -0.011, -0.032, -0.035, -0.122, -0.170,
        -0.192, -0.202, -0.247, -0.272, -0.284, -0.305, -0.317, -0.319
    };

    df->add_grouped_index_groups<double>("real_indexs11", real_s11);
    df->add_column<double>("PAE_contours",  pae);

    std::vector<double> pae_abs = pae;
    for (std::size_t i = 0; i < pae_abs.size(); ++i) {
        pae_abs[i] = std::fabs(pae_abs[i]);
    }

    auto df_abs = std::make_shared<exprdf::DataFrame>();
    df_abs->set_name("PAE | abs(PAE_contours)");
    df_abs->add_uniform_index<double>("level", {36.654, 41.654});
    df_abs->add_uniform_index<int>("number", {1});
    df_abs->add_grouped_index_groups<double>("real_indexs11", real_s11);
    df_abs->add_column<double>("PAE_contours", pae_abs);

    exprdf::DataFrameView view;
    view.setDataFrames({df, df_abs});
    view.setWindowTitle("PAE Contours — multi DataFrame view");
    view.resize(900, 700);
    view.show();

    return app.exec();
}
