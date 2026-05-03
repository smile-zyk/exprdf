#ifndef EXPRDF_QT_DATAFRAMEVIEW_H
#define EXPRDF_QT_DATAFRAMEVIEW_H

#include <QWidget>
#include <memory>
#include <vector>
#include <exprdf/exprdf.hpp>

class QTableView;
class QVBoxLayout;
class QTabWidget;

namespace exprdf {

class DataFrameModel;

class DataFrameView : public QWidget {
    Q_OBJECT
public:
    explicit DataFrameView(QWidget* parent = nullptr);

    void setDataFrame(const std::shared_ptr<DataFrame>& df);
    void setDataFrames(const std::vector<std::shared_ptr<DataFrame>>& dfs);
    std::shared_ptr<DataFrame> dataFrame() const;
    const std::vector<std::shared_ptr<DataFrame>>& dataFrames() const;

    DataFrameModel* model() const;
    DataFrameModel* model(std::size_t index) const;
    QTableView* tableView() const;
    QTableView* tableView(std::size_t index) const;
    int frameCount() const;

private:
    void clearTabs();
    void addDataFrameTab(const std::shared_ptr<DataFrame>& df, std::size_t index);
    QString tabNameFor(const std::shared_ptr<DataFrame>& df, std::size_t index) const;

    QTabWidget* tab_widget_;

    std::vector<std::shared_ptr<DataFrame>> dfs_;
    std::vector<QTableView*> table_views_;
    std::vector<DataFrameModel*> models_;
};

} // namespace exprdf

#endif // EXPRDF_QT_DATAFRAMEVIEW_H
