from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

def plot_history(H, epoch, metric):
      plt.figure(figsize=(6, 4))
      # plot training loss or accuracy
      plt.plot(np.arange(0, epoch), H.history[f'{metric}'], label=f'train_{metric}')
      # plot validation loss or accuracy
      plt.plot(np.arange(0, epoch), H.history[f'val_{metric}'], label=f'val_{metric}')
      plt.ylim((0, 1))
      plt.xlabel('Epochs')
      plt.ylabel('Loss' if metric=='Loss' else 'Accuracy')
      plt.legend()
      plt.grid()
      plt.show()


def print_evaluation_metrics(model, test_X, test_Y):
      predictions = model.predict(test_X, batch_size=32)
      # Convert probability vector into target vector
      predictions= np.where(predictions > 0.5, 1, 0)
      predictions.reshape(predictions.shape[0],)
      # print the metrics
      print("\nConfusion matrix: \n\n",metrics.confusion_matrix(test_Y, predictions))
      print('\n       -----------------     ')
      print('\n',classification_report(test_Y, predictions, target_names=['non smile','smile']))
      print('\n       -----------------     ')
      print("\nAUC: ",metrics.roc_auc_score(test_Y, predictions))

      
def print_metrics(model, test_X, test_Y):
      test_loss , test_accuracy , test_precision , test_recall = model.evaluate(x=test_X , y=test_y)
      print('\n test_loss = '+ str(test_loss) + 
           '\n test_accuracy = '+ str(test_accuracy) +
           '\n test_precision = '+ str(test_precision) +
           '\n test_recall = '+ str(test_recall))
