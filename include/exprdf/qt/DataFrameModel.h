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

    void setDataFrame(std::shared_ptr<DataFrame> df);
    std::shared_ptr<DataFrame> dataFrame() const;

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

    static const int FETCH_BATCH = 256;

    QString formatCell(const Column& col, std::size_t row) const;
    QString rowHeader(std::size_t row) const;
};

} // namespace exprdf

#endif // EXPRDF_QT_DATAFRAMEMODEL_H
