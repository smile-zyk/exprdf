#include <exprdf/qt/DataFrameModel.h>
#include <exprdf/UnitFormat.hpp>
#include <QString>

namespace exprdf {

DataFrameModel::DataFrameModel(QObject* parent)
    : QAbstractTableModel(parent)
    , fetched_rows_(0)
{
}

void DataFrameModel::setDataFrame(std::shared_ptr<DataFrame> df)
{
    beginResetModel();
    df_ = df;
    fetched_rows_ = 0;
    if (df_) {
        int total = static_cast<int>(df_->num_rows());
        fetched_rows_ = total < FETCH_BATCH ? total : FETCH_BATCH;
    }
    endResetModel();
}

std::shared_ptr<DataFrame> DataFrameModel::dataFrame() const
{
    return df_;
}

int DataFrameModel::rowCount(const QModelIndex& parent) const
{
    if (parent.isValid() || !df_) return 0;
    return fetched_rows_;
}

int DataFrameModel::columnCount(const QModelIndex& parent) const
{
    if (parent.isValid() || !df_) return 0;
    return static_cast<int>(df_->num_columns());
}

QVariant DataFrameModel::data(const QModelIndex& index, int role) const
{
    if (!df_ || !index.isValid()) return QVariant();
    if (role != Qt::DisplayRole) return QVariant();

    std::size_t row = static_cast<std::size_t>(index.row());
    std::size_t col_idx = static_cast<std::size_t>(index.column());
    if (row >= df_->num_rows() || col_idx >= df_->num_columns())
        return QVariant();

    const std::string& col_name = df_->column_name(col_idx);
    const Column& col = df_->get_column(col_name);
    return formatCell(col, row);
}

QVariant DataFrameModel::headerData(int section, Qt::Orientation orientation,
                                     int role) const
{
    if (!df_ || role != Qt::DisplayRole) return QVariant();

    if (orientation == Qt::Horizontal) {
        if (section < 0 || static_cast<std::size_t>(section) >= df_->num_columns())
            return QVariant();
        const std::string& name = df_->column_name(static_cast<std::size_t>(section));
        return QString::fromStdString(name);
    } else {
        // Vertical header: show multi-index if present, else row number
        if (section < 0 || static_cast<std::size_t>(section) >= df_->num_rows())
            return QVariant();
        return rowHeader(static_cast<std::size_t>(section));
    }
}

bool DataFrameModel::canFetchMore(const QModelIndex& parent) const
{
    if (parent.isValid() || !df_) return false;
    return fetched_rows_ < static_cast<int>(df_->num_rows());
}

void DataFrameModel::fetchMore(const QModelIndex& parent)
{
    if (parent.isValid() || !df_) return;

    int total = static_cast<int>(df_->num_rows());
    int remaining = total - fetched_rows_;
    int to_fetch = remaining < FETCH_BATCH ? remaining : FETCH_BATCH;

    if (to_fetch <= 0) return;

    beginInsertRows(QModelIndex(), fetched_rows_, fetched_rows_ + to_fetch - 1);
    fetched_rows_ += to_fetch;
    endInsertRows();
}

QString DataFrameModel::formatCell(const Column& col, std::size_t row) const
{
    const std::string& qty = col.quantity;

    switch (col.tag) {
    case DType::Int: {
        int v = col.as<int>()[row];
        if (!qty.empty())
            return QString::fromStdString(unit_format::Format(qty, v));
        return QString::number(v);
    }
    case DType::Double: {
        double v = col.as<double>()[row];
        if (!qty.empty())
            return QString::fromStdString(unit_format::Format(qty, v));
        return QString::number(v, 'g', 10);
    }
    case DType::String:
        return QString::fromStdString(col.as<std::string>()[row]);
    case DType::Complex: {
        const std::complex<double>& c = col.as<std::complex<double>>()[row];
        if (!qty.empty())
            return QString::fromStdString(unit_format::Format(qty, c));
        QString s;
        s += QString::number(c.real(), 'g', 6);
        if (c.imag() >= 0) s += " + ";
        else s += " - ";
        s += QString::number(std::abs(c.imag()), 'g', 6);
        s += " j";
        return s;
    }
    }
    return QString();
}

QString DataFrameModel::rowHeader(std::size_t row) const
{
    if (df_->num_indices() == 0) {
        return QString::number(static_cast<qlonglong>(row));
    }
    // Show per-dimension indices: "0,3" for multi_index result
    std::vector<std::size_t> mi = df_->multi_index(row);
    QString header;
    for (std::size_t i = 0; i < mi.size(); ++i) {
        if (i > 0) header += ',';
        header += QString::number(static_cast<qlonglong>(mi[i]));
    }
    return header;
}

} // namespace exprdf
