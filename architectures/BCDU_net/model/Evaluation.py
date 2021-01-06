from architectures.BCDU_net.model.Preprocessing import *
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

class Evaluation():
    def __init__(self):
        retina_blood_vessel_dataset = RetinaBloodVesselDataset()
        self.test_inputs, self.test_gt, self.test_bm = retina_blood_vessel_dataset.get_testing_data()
        self.evaluations_path = "architectures/BCDU_net/Tests/"
        self.y_pred = None
        self.y_true = None
        self.f1 = None
        self.roc = None
        self.conf_matrix = None
        self.new_pred = None
        self.acc = None
        self.sens = None
        self.spec = None


    def evaluation_data(self):
        """

        :return: Three numpy arrays, one for the test images, test groundtruth and test border masks
        """
        preprocessing = Preprocessing()
        test_prepro_inputs, test_prepro_bm = preprocessing.run_preprocess_pipeline(self.test_inputs, "test", self.test_gt)
        new_height, new_width = preprocessing.new_dimensions()
        test_prepro_inputs = np.einsum('klij->kijl', test_prepro_inputs)
        return test_prepro_inputs, test_prepro_bm, new_height, new_width



    def f1_score(self):
        """

        :return: The F1 score
        """
        f1 = f1_score(self.y_true, self.new_pred, labels=None, average='binary', sample_weight=None)
        print("\nF1 score (F-measure): " + str(f1))
        return f1

    def ROC(self):
        """

        :return: The ROC score
        """
        false_positives, true_positives, thresholds = sklearn.metrics.roc_curve((self.y_true), self.y_pred)
        AUC_ROC = roc_auc_score(self.y_true, self.y_pred)
        print("\nArea under the ROC curve: " + str(AUC_ROC))
        roc_curve = plt.figure()
        plt.plot(false_positives, true_positives, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_ROC)
        plt.title('ROC curve')
        plt.xlabel("FPR (False Positive Rate)")
        plt.ylabel("TPR (True Positive Rate)")
        plt.legend(loc="lower right")
        plt.savefig(self.evaluations_path + "50_ROC.png")
        return AUC_ROC


    def confusion_matrix(self):
        """

        Set the confusion matrix
        """
        threshold_confusion = 0.5
        print("\nConfusion matrix:  Custom threshold (for positive) of " + str(threshold_confusion))
        new_pred = np.empty((self.y_pred.shape[0]))
        for i in range(self.y_pred.shape[0]):
            if self.y_pred[i] >= threshold_confusion:
                new_pred[i] = 1
            else:
                new_pred[i] = 0
        confusion = confusion_matrix(self.y_true, new_pred)
        print(confusion)
        self.new_pred = new_pred
        self.conf_matrix = confusion

    def accuracy(self):
        """

        :return: The global accuracy of our model
        """
        sum = float(np.sum(self.conf_matrix))
        accuracy = float(self.conf_matrix[0,0] + self.conf_matrix[1,1])/sum
        print("Accuracy: " + str(accuracy))
        return accuracy

    def specificity(self):
        """

        :return: The specificity score
        """
        denomin = float(self.conf_matrix[0, 0] + self.conf_matrix[0, 1])
        specificity = float(self.conf_matrix[0, 0]) /denomin
        print("Specificity: " + str(specificity))
        return specificity

    def sensitivity(self):
        """

        :return: The sensitivity score
        """
        denom = float(self.conf_matrix[1, 1] + self.conf_matrix[1, 0])
        sensitivity = float(self.conf_matrix[1, 1]) /denom
        print("Sensitivity: " +str(sensitivity))
        return sensitivity

    def set_y_true_pred(self, y_true, y_pred):
        """

        :param y_true: The groundtruth values (real segmentations)
        :param y_pred: Our predictions on the segementations
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def evaluation_metrics(self, y_true, y_pred):
        """

        :param y_true: The groundtruth values (real segmentations)
        :param y_pred: Our predictions on the segementations
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.confusion_matrix()
        self.f1 = self.f1_score()
        self.roc = self.ROC()
        self.acc = self.accuracy()
        self.sens = self.sensitivity()
        self.spec = self.specificity()
        self.save_metrics_results()


    def save_metrics_results(self):
        """
        Save all the metrics' results in a file
        """
        # Save the results
        file_perf = open(self.evaluations_path + '50performances.txt', 'w')
        file_perf.write("Area under the ROC curve: " + str(self.roc)
                        + "\nF1 score (F-measure): " + str(self.f1)
                        + "\n\nConfusion matrix:"
                        + str(self.conf_matrix)
                        + "\nACCURACY: " + str(self.acc)
                        + "\nSENSITIVITY: " + str(self.sens)
                        + "\nSPECIFICITY: " + str(self.spec)
                        )
        file_perf.close()
