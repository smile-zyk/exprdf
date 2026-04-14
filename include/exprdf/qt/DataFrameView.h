#ifndef EXPRDF_QT_DATAFRAMEVIEW_H
#define EXPRDF_QT_DATAFRAMEVIEW_H

#include <QWidget>
#include <memory>
#include <exprdf/exprdf.hpp>

class QTableView;
class QLabel;

namespace exprdf {

class DataFrameModel;

class DataFrameView : public QWidget {
    Q_OBJECT
public:
    explicit DataFrameView(QWidget* parent = nullptr);

    void setDataFrame(std::shared_ptr<DataFrame> df);
    std::shared_ptr<DataFrame> dataFrame() const;

    DataFrameModel* model() const;
    QTableView* tableView() const;

private:
    QLabel* title_label_;
    QTableView* table_view_;
    DataFrameModel* model_;
};

} // namespace exprdf

#endif // EXPRDF_QT_DATAFRAMEVIEW_H
