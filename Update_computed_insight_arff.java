package DM;

import java.io.File;
import java.util.Arrays;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;

public class Update_computed_insight_arff {
    public static void main(String[] args) throws Exception {
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File("output/cleaned_computed_insight_success_of_active_sellers.arff"));
        Instances data = loader.getDataSet();

        Attribute urgencyTextRateGroup = new Attribute("urgency_text_rate_group",
                Arrays.asList("none", "low", "medium", "high"));
        data.insertAttributeAt(urgencyTextRateGroup, data.numAttributes());

        for (int i = 0; i < data.numInstances(); i++) {
            double val = data.instance(i).value(data.attribute("urgency_text_rate"));
            String group;
            if (val == 0) group = "none";
            else if (val <= 25) group = "low";
            else if (val <= 75) group = "medium";
            else group = "high";

            data.instance(i).setValue(data.attribute("urgency_text_rate_group"), group);
        }

        int idxUrgency = data.attribute("urgency_text_rate").index();
        int idxMerchant = data.attribute("merchant_id").index();
        data.deleteAttributeAt(Math.max(idxUrgency, idxMerchant));  
        data.deleteAttributeAt(Math.min(idxUrgency, idxMerchant));  

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("output/computed_insight_success_of_active_sellers_grouped.arff"));
        saver.writeBatch();
    }
}