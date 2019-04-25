from regression import LogRegModel
from data_processing import DataProcessor
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np

class Evaluator():
  def precision(self, conf_matrix):
    # tn fp
    # fn tp
    tn, fp, fn, tp = conf_matrix.ravel()
    return tp / (tp + fp)
  
  def recall(self, confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    return tp / (tp + fn)
  
  def f1_score(self, confusion_matrix):
    p = self.precision(confusion_matrix)
    r = self.recall(confusion_matrix)
    return (2 * p * r) / (p + r)

  def plot_roc(self, roc_curve, logit_roc_auc, model_name):
    plt.plot(roc_curve[0], roc_curve[1], label='{} (area = {:.4f})'.format(model_name, logit_roc_auc))
    plt.plot([0,1], [0,1],'r--')

  def roc_labels(self):
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

  def evaluate_k_fold(self, conf_matrices, coeffs, save_path):
    precisions = [self.precision(c) for c in conf_matrices]
    recalls = [self.recall(c) for c in conf_matrices]
    f1_scores = [self.f1_score(c) for c in conf_matrices]
    p_avg, p_std = np.average(precisions), np.std(precisions)
    r_avg, r_std = np.average(recalls), np.std(recalls)
    f1_avg, f1_std = np.average(f1_scores), np.std(f1_scores)

    conf = np.zeros((2,2))
    for c in conf_matrices:
      conf = np.add(conf, c)

    p_tot = self.precision(conf)
    r_tot = self.recall(conf)
    f1_tot = self.f1_score(conf)

    coeffs_avg = np.average(coeffs, axis=0)
    coeffs_std = np.std(coeffs, axis=0)

    with open(save_path + ".csv", "w") as f:
      f.write("{}\n".format(save_path))
      f.write(",{},{},{}\n".format("Average", "Std", "Total"))
      f.write("{},{},{},{}\n".format("Precision", p_avg, p_std, p_tot))
      f.write("{},{},{},{}\n".format("Recall", r_avg, r_std, r_tot))
      f.write("{},{},{},{}\n".format("F1 score", f1_avg, f1_std, f1_tot))
      
      f.write("{}\n{},{}\n{},{}\n".format(
        "Final confusion matrix",
        int(conf[0][0]), int(conf[0][1]), int(conf[1,0]), int(conf[1,1])))

      f.write("{}\n{},{}\n".format("Coefficients", "Average", "Std"))
      for c_avg, c_std in zip(coeffs_avg, coeffs_std):
        f.write("{},{}\n".format(c_avg, c_std))

  def compare_k_fold(self):
    path = "results/kfold_"
    dp = DataProcessor()
    dp.load('data/SQuAD/squad-v7.file')

    model = LogRegModel()
    model.load_vectors(dp.articles)
    result = model.run_k_fold()
    self.evaluate_k_fold(result["conf_matrices"], result["coefficients"], path + "sentiment_model")
    result = model.run_k_fold(with_sentiment=False)
    self.evaluate_k_fold(result["conf_matrices"], result["coefficients"], path + "baseline_model")



if __name__ == "__main__":
  ev = Evaluator()
  ev.compare_k_fold()
  # ev.compare_why_questions()