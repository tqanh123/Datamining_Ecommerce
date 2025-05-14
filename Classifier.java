package DataMining;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import java.io.File;
import java.util.Random;

public class J48&NaiveBayes {

    public static void main(String[] args) {
        try {
            // 1. Load the ARFF file
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File("filename.arff"));
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            // 2. Split the data into training and testing sets using Resample
            Resample resampleTrain = new Resample();
            resampleTrain.setSampleSizePercent(70);
            resampleTrain.setNoReplacement(true);
            resampleTrain.setRandomSeed(1);  // Fixed seed for reproducibility
            resampleTrain.setInputFormat(data);
            Instances trainData = Filter.useFilter(data, resampleTrain);

            Resample resampleTest = new Resample();
            resampleTest.setSampleSizePercent(30);
            resampleTest.setNoReplacement(true);
            resampleTest.setInvertSelection(true); // Use remaining data for testing
            resampleTest.setRandomSeed(1);  // Same seed for consistency
            resampleTest.setInputFormat(data);
            Instances testData = Filter.useFilter(data, resampleTest);

            System.out.println("Training instances: " + trainData.numInstances());
            System.out.println("Testing instances: " + testData.numInstances());

            // --- J48 Decision Tree ---
            System.out.println("\n--- J48 Decision Tree ---");
            J48 j48 = new J48();
            j48.buildClassifier(trainData);

            Evaluation evalJ48 = new Evaluation(trainData);
            evalJ48.evaluateModel(j48, testData);

            System.out.println(evalJ48.toSummaryString("J48 Results:\n======\n", false));
            System.out.println(evalJ48.toClassDetailsString("J48 Class Details:\n======\n"));
            System.out.println(evalJ48.toMatrixString("J48 Confusion Matrix:\n======\n"));

            // --- Naive Bayes ---
            System.out.println("\n--- Naive Bayes ---");
            NaiveBayes naiveBayes = new NaiveBayes();
            naiveBayes.buildClassifier(trainData);

            Evaluation evalNB = new Evaluation(trainData);
            evalNB.evaluateModel(naiveBayes, testData);

            System.out.println(evalNB.toSummaryString("Naive Bayes Results:\n======\n", false));
            System.out.println(evalNB.toClassDetailsString("Naive Bayes Class Details:\n======\n"));
            System.out.println(evalNB.toMatrixString("Naive Bayes Confusion Matrix:\n======\n"));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
