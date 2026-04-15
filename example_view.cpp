#include <QApplication>
#include <exprdf/exprdf.hpp>
#include <exprdf/UnitFormat.hpp>
#include <exprdf/qt/DataFrameView.h>
#include <cmath>

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    // 3D index: imag_index(20) x real_index(20) x freq(501) = 200,400 rows
    auto df = std::make_shared<exprdf::DataFrame>();
    df->set_name("Pdel");

    std::vector<double> imag_vals, real_vals, freq_vals;
    for (int i = 0; i < 20; ++i) imag_vals.push_back(-1.0 + 0.1 * i);
    for (int i = 0; i < 20; ++i) real_vals.push_back(0.05 + 0.05 * i);
    for (int i = 0; i <= 500; ++i) freq_vals.push_back(i * 0.02); // 0~10 GHz, step 20 MHz

    df->add_uniform_index<double>("imag_index", imag_vals);
    df->add_uniform_index<double>("real_index", real_vals);
    df->add_uniform_index<double>("freq", freq_vals);
    df->set_column_quantity("freq", unit_format::kFrequency);

    // Generate complex Pdel data: 200,400 rows
    std::size_t n = df->num_rows();
    std::vector<std::complex<double>> pdel;
    pdel.reserve(n);
    for (std::size_t r = 0; r < n; ++r) {
        auto idx = df->multi_index(r);
        double im_idx = imag_vals[idx[0]];
        double re_idx = real_vals[idx[1]];
        double f = freq_vals[idx[2]];
        double re = re_idx * std::cos(f * 0.5) * 0.1;
        double im = im_idx * std::sin(f * 0.3) * 0.05;
        pdel.push_back(std::complex<double>(re, im));
    }
    df->add_column<std::complex<double>>("Pdel", pdel, unit_format::kPower);

    exprdf::DataFrameView view;
    view.setDataFrame(df);
    view.setWindowTitle("DataFrame Viewer Example");
    view.resize(700, 500);
    view.show();

    return app.exec();
}
