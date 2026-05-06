#ifndef EXPRDF_QT_DATAFRAMEMODEL_H
#define EXPRDF_QT_DATAFRAMEMODEL_H

#include <QAbstractTableModel>
#include <memory>
#include <exprdf/exprdf.hpp>

namespace exprdf {

class DataFrameModel : public QAbstractTableModel {
    Q_OBJECT
public:
    explicit DataFrameModel(QObject* parent = nullptr);

    void setDataFrame(const std::shared_ptr<DataFrame>& df);
    std::shared_ptr<DataFrame> dataFrame() const;
    bool hasDataFrame() const;

    // QAbstractTableModel interface
    int rowCount(const QModelIndex& parent = QModelIndex()) const override;
    int columnCount(const QModelIndex& parent = QModelIndex()) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    QVariant headerData(int section, Qt::Orientation orientation,
                        int role = Qt::DisplayRole) const override;

    // fetchMore for lazy loading
    bool canFetchMore(const QModelIndex& parent) const override;
    void fetchMore(const QModelIndex& parent) override;

private:
    std::shared_ptr<DataFrame> df_;
    int fetched_rows_;

    // Pre-computed expanded column mapping: scalar cols map 1:1,
    // list/matrix cols expand to name(k) / name(i,j) sub-columns.
    struct ExpandedCol {
        std::string col_name;  // actual column name in df_
        std::string header;    // display header: "a", "a(1)", "M(2,3)", etc.
        std::size_t elem;      // 0-based flat element index; ~0 for scalar cols
    };
    std::vector<ExpandedCol> expanded_cols_;
    void rebuildExpandedCols();

    static const int FETCH_BATCH = 256;

    QString formatCell(const Column& col, std::size_t row, std::size_t elem) const;
    QString rowHeader(std::size_t row) const;
};

} // namespace exprdf

#endif // EXPRDF_QT_DATAFRAMEMODEL_H
