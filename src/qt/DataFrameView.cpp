#include <exprdf/qt/DataFrameView.h>
#include <exprdf/qt/DataFrameModel.h>
#include <QTableView>
#include <QLabel>
#include <QVBoxLayout>
#include <QHeaderView>
#include <QFont>

namespace exprdf {

DataFrameView::DataFrameView(QWidget* parent)
    : QWidget(parent)
{
    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(2);

    title_label_ = new QLabel(this);
    QFont font = title_label_->font();
    font.setBold(true);
    title_label_->setFont(font);
    title_label_->hide();
    layout->addWidget(title_label_);

    model_ = new DataFrameModel(this);

    table_view_ = new QTableView(this);
    table_view_->setModel(model_);
    table_view_->setAlternatingRowColors(true);
    table_view_->horizontalHeader()->setStretchLastSection(true);
    table_view_->verticalHeader()->setDefaultSectionSize(24);
    layout->addWidget(table_view_);
}

void DataFrameView::setDataFrame(std::shared_ptr<DataFrame> df)
{
    model_->setDataFrame(df);

    if (df && !df->name().empty()) {
        title_label_->setText(QString::fromStdString(df->name()));
        title_label_->show();
    } else {
        title_label_->hide();
    }
}

std::shared_ptr<DataFrame> DataFrameView::dataFrame() const
{
    return model_->dataFrame();
}

DataFrameModel* DataFrameView::model() const
{
    return model_;
}

QTableView* DataFrameView::tableView() const
{
    return table_view_;
}

} // namespace exprdf
