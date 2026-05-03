#include <exprdf/qt/DataFrameView.h>
#include <exprdf/qt/DataFrameModel.h>
#include <QTabWidget>
#include <QTableView>
#include <QWidget>
#include <QVBoxLayout>
#include <QHeaderView>

namespace exprdf {

DataFrameView::DataFrameView(QWidget* parent)
    : QWidget(parent)
    , tab_widget_(nullptr)
{
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(2);

    tab_widget_ = new QTabWidget(this);
    layout->addWidget(tab_widget_);
}

void DataFrameView::setDataFrame(const std::shared_ptr<DataFrame>& df)
{
    setDataFrames(df ? std::vector<std::shared_ptr<DataFrame> >(1, df)
                     : std::vector<std::shared_ptr<DataFrame> >());
}

void DataFrameView::setDataFrames(const std::vector<std::shared_ptr<DataFrame>>& dfs)
{
    clearTabs();

    dfs_.clear();
    table_views_.clear();
    models_.clear();

    for (std::size_t i = 0; i < dfs.size(); ++i) {
        if (dfs[i]) {
            dfs_.push_back(dfs[i]);
        }
    }

    for (std::size_t i = 0; i < dfs_.size(); ++i) {
        addDataFrameTab(dfs_[i], i);
    }
}

std::shared_ptr<DataFrame> DataFrameView::dataFrame() const
{
    if (dfs_.empty()) return std::shared_ptr<DataFrame>();
    return dfs_.front();
}

const std::vector<std::shared_ptr<DataFrame>>& DataFrameView::dataFrames() const
{
    return dfs_;
}

DataFrameModel* DataFrameView::model() const
{
    return model(0);
}

DataFrameModel* DataFrameView::model(std::size_t index) const
{
    if (index >= models_.size()) return nullptr;
    return models_[index];
}

QTableView* DataFrameView::tableView() const
{
    return tableView(0);
}

QTableView* DataFrameView::tableView(std::size_t index) const
{
    if (index >= table_views_.size()) return nullptr;
    return table_views_[index];
}

int DataFrameView::frameCount() const
{
    return static_cast<int>(dfs_.size());
}

void DataFrameView::clearTabs()
{
    while (tab_widget_->count() > 0) {
        QWidget* page = tab_widget_->widget(0);
        tab_widget_->removeTab(0);
        delete page;
    }
}

void DataFrameView::addDataFrameTab(const std::shared_ptr<DataFrame>& df, std::size_t index)
{
    QWidget* page = new QWidget(tab_widget_);
    QVBoxLayout* page_layout = new QVBoxLayout(page);
    page_layout->setContentsMargins(0, 0, 0, 0);
    page_layout->setSpacing(0);

    DataFrameModel* model = new DataFrameModel(page);
    model->setDataFrame(df);

    QTableView* table_view = new QTableView(page);
    table_view->setModel(model);
    table_view->setAlternatingRowColors(true);
    table_view->horizontalHeader()->setStretchLastSection(false);
    table_view->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    table_view->verticalHeader()->setDefaultSectionSize(24);
    table_view->resizeColumnsToContents();
    page_layout->addWidget(table_view);

    tab_widget_->addTab(page, tabNameFor(df, index));
    table_views_.push_back(table_view);
    models_.push_back(model);
}

QString DataFrameView::tabNameFor(const std::shared_ptr<DataFrame>& df, std::size_t index) const
{
    if (df && !df->name().empty()) {
        return QString::fromStdString(df->name());
    }

    if (df && df->num_columns() > 0) {
        std::size_t last = df->num_columns() - 1;
        return QString::fromStdString(df->column_name(last));
    }

    return QString("DataFrame %1").arg(static_cast<int>(index + 1));
}

} // namespace exprdf
