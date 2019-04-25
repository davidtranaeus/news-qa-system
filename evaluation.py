from regression import LogRegModel
from data_processing import DataProcessor
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from pprint import pprint

class Evaluator():
  def score(self, model):
    return model.model.score(
      model.test_vectors,
      model.test_targets
    )

  def confusion_matrix(self, model):
    # tn fp
    # fn tp
    return confusion_matrix(
      model.model.predict(model.test_vectors),
      model.test_targets
    )

  def precision(self, conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    return tp / (tp + fp)
  
  def recall(self, confusion_matrix):
    tn, fp, fn, tp = confusion_matrix.ravel()
    return tp / (tp + fn)
  
  def f1_score(self, confusion_matrix):
    p = self.precision(confusion_matrix)
    r = self.recall(confusion_matrix)
    return (2 * p * r) / (p + r)

  def coefficients(self, model):
    return model.model.coef_
  
  def n_iterations(self, model):
    return model.model.n_iter_

  def logit_roc_auc(self, model):
    return roc_auc_score(
      model.test_targets, 
      model.model.predict(model.test_vectors))
  
  def roc_curve(self, model):
    return roc_curve(
      model.test_targets, 
      model.model.predict_proba(model.test_vectors)[:,1])

  def plot_roc(self, roc_curve, logit_roc_auc, model_name):
    plt.plot(roc_curve[0], roc_curve[1], label='{} (area = {:.2f})'.format(model_name, logit_roc_auc))
    plt.plot([0,1], [0,1],'r--')

  def roc_labels(self):
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

  def evaluate_model(self, model, save_path):
    with open(save_path + ".csv", "w") as f:
      f.write("{}\n".format(save_path))
      f.write("{},{}\n".format("Score", self.score(model)))
      conf = self.confusion_matrix(model)
      # print(conf)
      f.write("{},{}\n".format("Precision", self.precision(conf)))
      f.write("{},{}\n".format("Recall", self.recall(conf)))
      f.write("{},{}\n".format("F1 score", self.f1_score(conf)))
      
      f.write("{}\n{},{}\n{},{}\n".format(
        "Confusion matrix",
        conf[0][0], conf[0][1], conf[1,0], conf[1,1]))

      f.write("Coefficients\n")
      for c in self.coefficients(model)[0]:
        f.write("{},".format(c))

  ####################################

  def compare_sentiment_with_baseline(self):
    path = "results/"
    dp = DataProcessor()
    dp.load('data/SQuAD/squad-v7.file')

    model = LogRegModel()
    model.load_vectors(dp.articles, with_sentiment=False)
    model.train()
    self.evaluate_model(model, path + "baseline_model")
    self.plot_roc(
      self.roc_curve(model),
      self.logit_roc_auc(model),
      "Baseline model")

    model.load_vectors(dp.articles)
    model.train()
    self.evaluate_model(model, path + "sentiment_model")
    self.plot_roc(
      self.roc_curve(model),
      self.logit_roc_auc(model),
      "Sentiment model")

    self.roc_labels()

    plt.savefig('results/roc_baseline_sentiment')
    # plt.show()

  def compare_why_questions(self):
    dp = DataProcessor()
    dp.load('data/SQuAD/squad-v7.file')

    why_ids = []
    for a_idx, art in enumerate(dp.articles):
      for q_idx, question in enumerate(art["questions"]):
        if question["question"]["tokens"][0].lower() == 'why':
          why_ids.append([a_idx, q_idx])

    path = "results/why_questions_"

    model = LogRegModel()
    model.load_vectors(dp.articles, with_sentiment=False)
    model.train(why_ids)
    self.evaluate_model(model, path + "baseline_model")
    self.plot_roc(
      self.roc_curve(model),
      self.logit_roc_auc(model),
      "Baseline model")
    model.load_vectors(dp.articles)
    model.train(why_ids)
    self.evaluate_model(model, path + "sentiment_model")
    self.plot_roc(
      self.roc_curve(model),
      self.logit_roc_auc(model),
      "Sentiment model")

    self.roc_labels()

    plt.savefig('results/roc_why_baseline_sentiment')


if __name__ == "__main__":
  ev = Evaluator()
  # ev.compare_sentiment_with_baseline()
  ev.compare_why_questions()