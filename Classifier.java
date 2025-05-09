package DataMining;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import java.util.Random;

public class Classifier {
    public static void main(String[] args) throws Exception {
        // Read Data
        DataSource source = new DataSource("computed_insight_success_of_active_sellers.arff");
        Instances data = source.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // ==== J48 ====
        System.out.println("=== J48 ===");
        long startJ48 = System.currentTimeMillis();
        J48 j48 = new J48();
        Evaluation evalJ48 = new Evaluation(data);
        evalJ48.crossValidateModel(j48, data, 10, new Random(1));
        long endJ48 = System.currentTimeMillis();

        System.out.println(evalJ48.toSummaryString());
        System.out.println(evalJ48.toClassDetailsString()); // Precision, Recall, F-measure
        System.out.println(evalJ48.toMatrixString());       // Confusion Matrix
        System.out.println("Runtime: " + (endJ48 - startJ48) + " ms\n");

        // ==== NaiveBayes ====
        System.out.println("=== NaiveBayes ===");
        long startNB = System.currentTimeMillis();
        NaiveBayes nb = new NaiveBayes();
        Evaluation evalNB = new Evaluation(data);
        evalNB.crossValidateModel(nb, data, 10, new Random(1));
        long endNB = System.currentTimeMillis();

        System.out.println(evalNB.toSummaryString());
        System.out.println(evalNB.toClassDetailsString());
        System.out.println(evalNB.toMatrixString());
        System.out.println("Runtime: " + (endNB - startNB) + " ms");
    }
}
