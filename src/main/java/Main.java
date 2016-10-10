import com.google.common.base.Joiner;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;


public class Main {

    public static final String RootDir = "../";
    public static final String DataDir = RootDir + "data/";
    public static final String ModelDir = RootDir + "model/";
    public static final String ResultDir = RootDir + "result/";

    public static final String TrainFile = DataDir + "train.arff";
    public static final String TestFile = DataDir + "test.arff";

    public static void main(String[] args) {

        List<MyClassifier> classifiers = new ArrayList<>();

        classifiers.add(Repo.Classifiers.LOG_RF_SMO);

        try {

            //Read train instances
            DataSource trainSource = new DataSource(TrainFile);
            Instances trainData = trainSource.getDataSet();

            //Read test instances
            DataSource testSource = new DataSource(TestFile);
            Instances testData = testSource.getDataSet();

            //Set the last feature as the class value
            trainData.setClassIndex(trainData.numAttributes() - 1);
            testData.setClassIndex(testData.numAttributes() - 1);

            //Produce labeled test data for each classifier
            for (MyClassifier classifier : classifiers) {
                BuildModelAndLabelTestData(classifier.classifier, classifier.Id, trainData, testData);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * Trains a classifier and creates a file which contains class probabilities for each test instance at a line.
     */
    static void BuildModelAndLabelTestData(Classifier classifier, String modelId, Instances train, Instances test) throws Exception {

        //build model with training data
        classifier.buildClassifier(train);

        //Save model
        String modelFile = ModelDir + modelId + ".model";
        SerializationHelper.write(modelFile, classifier);

        //Create kaggle CSV file for results
        BufferedWriter writer = new BufferedWriter(new FileWriter(ResultDir + modelId + ".csv"));
        writer.write("id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9");
        writer.newLine();

        //For each instance get the probabiity distribution ovel classes and add a new line to the file.
        for (int i = 0; i < test.numInstances(); i++) {

            Instance currentInstance = test.instance(i);

            double[] distribution = classifier.distributionForInstance(currentInstance);

            int instanceId = i + 1;

            String line = toString(instanceId, distribution);

            writer.write(line);
            writer.newLine();
        }

        //Close the file
        writer.flush();
        writer.close();
    }

    /**
     * returns a CSV line which contains the id of the instance and the probability distribution over all classes for the instance.
     */
    static String toString(int id, double[] distribution) {
        return id + "," + Joiner.on(",")
                .join(Arrays.stream(distribution).boxed()
                        .collect(Collectors.toList()));
    }

    /**
     * returns a CSV line which contains the id of the instance and the value 1 for the most probable class and 0 for others
     */
    static String toString(int id, int index) {
        int arr[] = new int[10];
        arr[0] = id;
        arr[index + 1] = 1;

        return Joiner.on(",")
                .join(Arrays.stream(arr).boxed()
                        .collect(Collectors.toList()));
    }

    /**
     * Find the index of most probable class in distribution.
     */
    static int max(double[] distribution) {
        int maxIndex = -1;
        double max = Double.MIN_VALUE;
        for (int i = 0; i < distribution.length; i++) {
            double v = distribution[i];
            if (v > max) {
                max = v;
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
