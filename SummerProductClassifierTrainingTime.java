import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import java.io.File;
import java.util.Random;

public class SummerProductClassifierTrainingTime {

    public static void main(String[] args) {
        try {
            // Load data and split (same as before)
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File("/Users/nhatminhnguyen/Desktop/Dataming_project/Dataset/DataSetInArff/summer_products_rating_group_label.arff"));
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            int trainSize = (int) Math.round(data.numInstances() * 0.7);
            int testSize = data.numInstances() - trainSize;

            Resample resampleTrain = new Resample();
            resampleTrain.setSampleSizePercent(70);
            resampleTrain.setNoReplacement(true);
            resampleTrain.setRandomSeed(new Random(1).nextInt());
            resampleTrain.setInputFormat(data);
            Instances trainData = Filter.useFilter(data, resampleTrain);

            Resample resampleTest = new Resample();
            resampleTest.setSampleSizePercent(30);
            resampleTest.setNoReplacement(true);
            resampleTest.setInvertSelection(true);
            resampleTest.setRandomSeed(new Random(1).nextInt());
            resampleTest.setInputFormat(data);
            Instances testData = Filter.useFilter(data, resampleTest);

            // --- k-Nearest Neighbors (IBk) ---
            System.out.println("\n--- k-Nearest Neighbors (IBk) ---");
            IBk ibk = new IBk();
            long startTimeIBk = System.currentTimeMillis();
            ibk.buildClassifier(trainData);
            long endTimeIBk = System.currentTimeMillis();
            double trainingTimeIBk = (endTimeIBk - startTimeIBk) / 1000.0; // in seconds

            Evaluation evalIbk = new Evaluation(trainData);
            evalIbk.evaluateModel(ibk, testData);

            System.out.println(evalIbk.toSummaryString("IBk Results:\n======\n", false));
            System.out.println("IBk Training Time: " + trainingTimeIBk + " seconds");

            // --- RandomForest ---
            System.out.println("\n--- RandomForest ---");
            RandomForest randomForest = new RandomForest();
            long startTimeRF = System.currentTimeMillis();
            randomForest.buildClassifier(trainData);
            long endTimeRF = System.currentTimeMillis();
            double trainingTimeRF = (endTimeRF - startTimeRF) / 1000.0; // in seconds

            Evaluation evalRF = new Evaluation(trainData);
            evalRF.evaluateModel(randomForest, testData);

            System.out.println(evalRF.toSummaryString("RandomForest Results:\n======\n", false));
            System.out.println("RandomForest Training Time: " + trainingTimeRF + " seconds");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}