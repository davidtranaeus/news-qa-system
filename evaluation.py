from regression import LogRegModel
from data_processing import DataProcessor
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
import matplotlib.patches as mpatches
import pickle

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

  def run_k_folds(self, n_runs=5, n_folds=2):
    dp = DataProcessor()
    dp.load('data/SQuAD/squad-v7.file')
    model = LogRegModel()
    model.load_vectors(dp.articles, n_folds=n_folds)
    
    baseline_results = []
    sentiment_results = []

    for run in range(n_runs):
      print("k-fold run:", run)
      baseline_results.append(model.run_k_fold(with_sentiment=False))
      sentiment_results.append(model.run_k_fold())
      model.create_new_folds(n_folds=n_folds)

    self.save_results(baseline_results, "results/5x2_baseline")
    self.save_results(sentiment_results, "results/5x2_sentiment")

  def run_5x2cv_paired_t_tests(self):
    baseline_results = self.load_results("results/5x2_baseline")
    sentiment_results = self.load_results("results/5x2_sentiment")

    variances = {
      "precision": [],
      "recall": [],
      "f1": []
    }

    metrics = list(variances.keys())

    functions = [
      self.precision,
      self.recall,
      self.f1_score
    ]

    ts = []

    for i in range(5):
      for f, met in zip(functions, metrics):
        # performance diff set 1 
        p1 = f(baseline_results[i]["conf_matrices"][0]) - f(sentiment_results[i]["conf_matrices"][0])

        # performance diff set 2
        p2 = f(baseline_results[i]["conf_matrices"][1]) - f(sentiment_results[i]["conf_matrices"][1])

        # diff mean
        mean = (p1 + p2) / 2
        
        # diff variance
        variances[met].append(np.square(p1 - mean) + np.square(p2 - mean))

    for i in range(len(metrics)):
      # first diff mean
      p = functions[i](baseline_results[0]["conf_matrices"][0]) - functions[i](sentiment_results[0]["conf_matrices"][0])

      # t statistic
      ts.append(p / np.sqrt((1/5) * np.sum(variances[metrics[i]])))

    print(ts)

  def avg_stats_from_k_folds(self, result_path, save_path):
    results = self.load_results(result_path)
    precisions = []
    recalls = []
    f1_scores = []
    coeffs = []
    aucs = []

    for run in results:

      # combine matrices from both folds
      mat = np.zeros((2,2))
      for c in run["conf_matrices"]:
        mat += c
      
      precisions.append(self.precision(mat))
      recalls.append(self.recall(mat))
      f1_scores.append(self.f1_score(mat))

      coeffs.append(np.average(run["coefficients"], axis=0))
      aucs.append(np.average(run["roc_auc_scores"]))
    
    p_avg, p_std = np.average(precisions), np.std(precisions)
    r_avg, r_std = np.average(recalls), np.std(recalls)
    f1_avg, f1_std = np.average(f1_scores), np.std(f1_scores)
    auc_avg, auc_std = np.average(aucs), np.std(aucs)
    coef_avg = np.average(coeffs, axis=0)
    coef_std = np.std(coeffs, axis=0)

    with open(save_path + ".csv", "w") as f:
      f.write("{}\n".format(save_path))
      f.write(",{},{}\n".format("Mean", "SD"))
      f.write("{},{},{}\n".format("Precision", p_avg, p_std))
      f.write("{},{},{}\n".format("Recall", r_avg, r_std))
      f.write("{},{},{}\n".format("F1 score", f1_avg, f1_std))
      
      f.write("{}\n{},{}\n".format("Coefficients", "Mean", "SD"))
      for c_avg, c_std in zip(coef_avg, coef_std):
        f.write("{},{}\n".format(c_avg, c_std))

      f.write("{},{},{}\n".format("AUC", "Mean", "SD"))
      f.write(",{},{}\n".format(auc_avg, auc_std))

  def plot_error_bars(self):
    base_y, base_std = [], []
    base_coef, base_coef_std = [], []

    with open('results/5x2_baseline_stats.csv', 'r') as f:
      for idx, row in enumerate(f.readlines()):
        if idx in range(2,5):
          vals = row.strip().split(',')
          base_y.append(float(vals[1]))
          base_std.append(float(vals[2]))

        if idx in range(7,11):
          vals = row.strip().split(',')
          base_coef.append(float(vals[0]))
          base_coef_std.append(float(vals[1]))

    sent_y, sent_std = [], []
    sent_coef, sent_coef_std = [], []

    with open('results/5x2_sentiment_stats.csv', 'r') as f:
      for idx, row in enumerate(f.readlines()):
        if idx in range(2,5):
          vals = row.strip().split(',')
          sent_y.append(float(vals[1]))
          sent_std.append(float(vals[2]))

        if idx in range(7,15):
          vals = row.strip().split(',')
          sent_coef.append(float(vals[0]))
          sent_coef_std.append(float(vals[1]))

    # Result, precision, F1 score
    plt.figure(0)
    plt.errorbar(list(range(3)), base_y, base_std, linestyle='None', capsize=5, ecolor='orange')
    plt.errorbar(list(range(3)), sent_y, sent_std, linestyle='None', capsize=8, ecolor='dodgerblue')
    ora_patch = mpatches.Patch(color='orange', label='Baseline model')
    blue_patch = mpatches.Patch(color='dodgerblue', label='Sentiment model')
    plt.legend(handles=[blue_patch, ora_patch])
    plt.xlabel('Metric')
    plt.ylabel('Result')
    plt.xticks(list(range(3)), ["Precision", "Recall", "F1 score"], rotation='45')
    plt.tight_layout()
    plt.savefig("results/final_metrics.svg")

    # # Coefficients 1-4
    # plt.figure(1)
    # plt.errorbar(list(range(4)), base_coef, base_coef_std, linestyle='None', capsize=5, ecolor='orange')
    # plt.errorbar(list(range(4)), sent_coef[:4], sent_coef_std[:4], linestyle='None', capsize=8, ecolor='dodgerblue')
    # ora_patch = mpatches.Patch(color='orange', label='Baseline model')
    # blue_patch = mpatches.Patch(color='dodgerblue', label='Sentiment model')
    # plt.legend(handles=[blue_patch, ora_patch])
    # plt.xlabel('Feature number')
    # plt.ylabel('Coefficient value')
    # plt.xticks(list(range(4)), ["1", "2", "3", "4"], rotation='45')
    # plt.tight_layout()
    # plt.savefig("results/first_coefficients.svg")

    # # Coefficients 5-8
    # plt.figure(2)
    # plt.errorbar(list(range(4)), sent_coef[4:], sent_coef_std[4:], linestyle='None', capsize=8, ecolor='dodgerblue')
    # blue_patch = mpatches.Patch(color='dodgerblue', label='Sentiment model')
    # plt.legend(handles=[blue_patch])
    # plt.xlabel('Feature number')
    # plt.ylabel('Coefficient value')
    # plt.xticks(list(range(4)), ["5", "6", "7", "8"], rotation='45')
    # plt.tight_layout()
    # plt.savefig("results/last_coefficients.svg")

    # All coefficients
    plt.figure(3)
    plt.errorbar(list(range(4)), base_coef, base_coef_std, linestyle='None', capsize=5, ecolor='orange')
    plt.errorbar(list(range(8)), sent_coef, sent_coef_std, linestyle='None', capsize=8, ecolor='dodgerblue')
    plt.plot(list(range(8)), [0 for i in range(8)], color='r', ls='--', dashes=(5, 20), linewidth=1)
    ora_patch = mpatches.Patch(color='orange', label='Baseline model (coefficients 1 to 4)')
    blue_patch = mpatches.Patch(color='dodgerblue', label='Sentiment model (coefficients 1 to 8)')
    plt.legend(handles=[blue_patch, ora_patch])
    plt.xlabel('Coefficient number')
    plt.ylabel('Coefficient value')
    plt.xticks(list(range(8)), [str(i+1) for i in range(8)], rotation='45')
    plt.tight_layout()
    plt.savefig("results/all_coefficients.svg")



  def save_results(self, results, file_path):
    with open(file_path + ".file", "wb") as f:
      pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
  
  def load_results(self, file_path):
    with open(file_path + ".file", "rb") as f:
      return pickle.load(f)


if __name__ == "__main__":
  ev = Evaluator()
  ev.plot_error_bars()
  ev.run_5x2cv_paired_t_tests()