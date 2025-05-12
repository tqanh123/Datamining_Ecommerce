
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

public class ComputedInsightSuccessOfActiveSeller {

    public static void main(String[] args) {
        try {
            // 1. Load the ARFF file
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File("/Users/nhatminhnguyen/Desktop/Dataming_project/Dataset/DataSetInArff/computed_insight_success_of_active_sellers.arff")); // Replace with the actual path
            Instances data = loader.getDataSet();

            // 2. Set the class attribute (assuming 'urgency_text_rate_group' is the last attribute)
            data.setClassIndex(data.numAttributes() - 1);

            // 3. Split the data into training and testing sets
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

            System.out.println("Training instances: " + trainData.numInstances());
            System.out.println("Testing instances: " + testData.numInstances());

            // --- k-Nearest Neighbors (IBk) ---
            System.out.println("\n--- k-Nearest Neighbors (IBk) ---");
            Classifier ibk = new IBk();
            ibk.buildClassifier(trainData);

            Evaluation evalIbk = new Evaluation(trainData);
            evalIbk.evaluateModel(ibk, testData);

            System.out.println(evalIbk.toSummaryString("IBk Results:\n======\n", false));
            System.out.println(evalIbk.toClassDetailsString("IBk Class Details:\n======\n"));
            System.out.println(evalIbk.toMatrixString("IBk Confusion Matrix:\n======\n"));

            // --- RandomForest ---
            System.out.println("\n--- RandomForest ---");
            Classifier randomForest = new RandomForest();
            randomForest.buildClassifier(trainData);

            Evaluation evalRF = new Evaluation(trainData);
            evalRF.evaluateModel(randomForest, testData);

            System.out.println(evalRF.toSummaryString("RandomForest Results:\n======\n", false));
            System.out.println(evalRF.toClassDetailsString("RandomForest Class Details:\n======\n"));
            System.out.println(evalRF.toMatrixString("RandomForest Confusion Matrix:\n======\n"));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}