from flask import Flask, render_template
from src import land, lang, rlgn

app = Flask(__name__)

@app.route("/")
def hello():
  land_train_score, land_train_acc, land_test_score, land_test_acc, land_grid_result_score, land_grid_result_params = land.landNN()
  land_DT_acc, land_DT_class_report, land_DT_conf_matrix = land.landDT()
  land_SVM_class_report = land.landSVM()

  lang_train_score, lang_train_acc, lang_test_score, lang_test_acc, lang_grid_result_score, lang_grid_result_params = lang.langNN()
  lang_DT_acc, lang_DT_class_report, lang_DT_conf_matrix = lang.langDT()
  lang_SVM_class_report = lang.langSVM()

  return render_template('./index.html', 
    land_train_score=land_train_score, 
    land_train_acc=land_train_acc, 
    land_test_score=land_test_score, 
    land_test_acc=land_test_acc,
    land_grid_result_score=land_grid_result_score, 
    land_grid_result_params=land_grid_result_params,
    land_DT_acc=land_DT_acc,
    land_DT_conf_matrix=land_DT_conf_matrix,
    titles=[''],
    land_DT_tables=[land_DT_class_report.to_html()],
    land_SVM_tables=[land_SVM_class_report.to_html()],
    lang_train_score=lang_train_score, 
    lang_train_acc=lang_train_acc, 
    lang_test_score=lang_test_score, 
    lang_test_acc=lang_test_acc,
    lang_grid_result_score=lang_grid_result_score, 
    lang_grid_result_params=lang_grid_result_params,
    lang_DT_acc=lang_DT_acc,
    lang_DT_conf_matrix=lang_DT_conf_matrix,
    lang_DT_tables=[lang_DT_class_report.to_html()],
    lang_SVM_tables=[lang_SVM_class_report.to_html()],
    )

if __name__ == "__main__":
  app.run()