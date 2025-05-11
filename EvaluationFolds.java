package DataMining;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.Random;

public class EvaluationFolds {
    public static void main(String[] args) throws Exception {
        DataSource source = new DataSource("data/summer_products_rating_group_label.arff");
        Instances dataset = source.getDataSet();
        dataset.setClassIndex(dataset.numAttributes() - 1);

        Classifier[] classifiers = {
                new NaiveBayes(),
                new IBk(5), // k=5 for IBk
                new RandomForest(),
                new J48()
        };
        String[] classifierNames = {"NaiveBayes", "IBk", "RandomForest", "J48"};

        int seed = 1;
        int folds = 10;
        Random rand = new Random(seed);
        Instances randData = new Instances(dataset);
        randData.randomize(rand);
        if (randData.classAttribute().isNominal()) {
            randData.stratify(folds);
        }

        // File to store results
        FileWriter writer = new FileWriter("results/evaluation_results.txt");
        DecimalFormat df = new DecimalFormat("#.###");

        for (int i = 0; i < classifiers.length; i++) {
            Classifier model = classifiers[i];
            String classifierName = classifierNames[i];
            writer.write("\n=== Evaluation for " + classifierName + " ===\n");

            Evaluation eval = new Evaluation(randData);
            double totalBuildTime = 0;
            double totalPredictTime = 0;

            // Perform 10-fold cross-validation
            for (int n = 0; n < folds; n++) {
                Instances train = randData.trainCV(folds, n);
                Instances test = randData.testCV(folds, n);

                // Measure build time
                long startBuild = System.nanoTime();
                model.buildClassifier(train);
                long endBuild = System.nanoTime();
                double buildTime = (endBuild - startBuild) / 1_000_000.0;
                totalBuildTime += buildTime;

                // Measure prediction time
                long startPredict = System.nanoTime();
                eval.evaluateModel(model, test);
                long endPredict = System.nanoTime();
                double predictTime = (endPredict - startPredict) / 1_000_000.0;
                totalPredictTime += predictTime;

                // Output fold results
                writer.write("\nFold " + (n + 1) + ":\n");
                writer.write("Build Time: " + df.format(buildTime) + " ms\n");
                writer.write("Prediction Time: " + df.format(predictTime) + " ms\n");
                writer.write(eval.toMatrixString("Confusion Matrix:\n"));
                writer.write("Accuracy: " + df.format(eval.pctCorrect()) + "%\n");
                writer.write("Precision: " + df.format(eval.precision(1)) + "\n");
                writer.write("Recall: " + df.format(eval.recall(1)) + "\n");
                writer.write("F-Measure: " + df.format(eval.fMeasure(1)) + "\n");
            }

            // Summary statistics
            writer.write("\nSummary for " + classifierName + ":\n");
            writer.write("Average Build Time: " + df.format(totalBuildTime / folds) + " ms\n");
            writer.write("Average Prediction Time: " + df.format(totalPredictTime / folds) + " ms\n");
            writer.write("Overall Accuracy: " + df.format(eval.pctCorrect()) + "%\n");
            writer.write("Overall Precision: " + df.format(eval.weightedPrecision()) + "\n");
            writer.write("Overall Recall: " + df.format(eval.weightedRecall()) + "\n");
            writer.write("Overall F-Measure: " + df.format(eval.weightedFMeasure()) + "\n");
            writer.write(eval.toSummaryString("Detailed Summary:\n", false));
        }

        writer.close();
        System.out.println("Evaluation results saved to evaluation_results.txt");
    }
}