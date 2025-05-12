import weka.classifiers.functions.SimpleLinearRegression;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.classifiers.Evaluation;
import java.io.File;

public class UniqueCategoriesSortedByCount {

    public static void main(String[] args) {
        try {
            // 1. Load the ARFF file
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File("/Users/nhatminhnguyen/Desktop/Dataming_project/Dataset/DataSetInArff/unique-categories.sorted-by-count.arff"));
            Instances data = loader.getDataSet();
            data.setClassIndex(data.numAttributes() - 1); // 'count' is the last

            // 2. Convert the 'keyword' attribute (index 0) to binary
            NominalToBinary nomToBin = new NominalToBinary();
            nomToBin.setAttributeIndices("1"); // Convert the first attribute (keyword)
            nomToBin.setInputFormat(data);
            Instances dataBinary = Filter.useFilter(data, nomToBin);

            // 3. Split the binary data
            int trainSize = (int) Math.round(dataBinary.numInstances() * 0.7);
            Instances trainData = new Instances(dataBinary, 0, trainSize);
            Instances testData = new Instances(dataBinary, trainSize, dataBinary.numInstances() - trainSize);

            System.out.println("Training instances (binary features): " + trainData.numInstances());
            System.out.println("Testing instances (binary features): " + testData.numInstances());

            // --- Simple Linear Regression ---
            System.out.println("\n--- Simple Linear Regression ---");
            SimpleLinearRegression slrRegressor = new SimpleLinearRegression();
            slrRegressor.buildClassifier(trainData);

            Evaluation evalSLRReg = new Evaluation(trainData);
            evalSLRReg.evaluateModel(slrRegressor, testData);

            System.out.println(evalSLRReg.toSummaryString("Simple Linear Regression Results:\n======\n", false));
            System.out.println("Simple Linear Regression Correlation Coefficient: " + evalSLRReg.correlationCoefficient());
            System.out.println("Simple Linear Regression Mean Absolute Error: " + evalSLRReg.meanAbsoluteError());
            System.out.println("Simple Linear Regression Root Mean Squared Error: " + evalSLRReg.rootMeanSquaredError());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}