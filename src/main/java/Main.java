import com.google.common.base.Joiner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
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

        classifiers.add(Repo.Classifiers.EnsembleMetaLogisticWithSvmNormKernAndRandomForest);

        //ReadInstances train instances
        Instances trainData = ReadInstances(TrainFile);

        //ReadInstances test instances
        Instances testData = ReadInstances(TestFile);

        //Produce labeled test data for each classifier
        for (MyClassifier classifier : classifiers) {
            try {
                BuildModelAndLabelTestData(classifier.classifier, classifier.Id, trainData, testData);
            } catch (Exception e) {
                System.out.println("Something unexpected has happened with " + classifier.Id + ". Continuing with the next classifier.");
                e.printStackTrace();
            }
        }
    }

    /**
     * Trains a classifier and creates a file which contains class probabilities for each test instance at a line.
     */
    static void BuildModelAndLabelTestData(Classifier classifier, String modelId, Instances train, Instances test) throws Exception {

        System.out.println("===[ " + modelId + " ]====================================");

        System.out.println("Training classifier...\n");
        classifier.buildClassifier(train);

        System.out.println("Saving model...\n");
        String modelFile = ModelDir + modelId + ".model";
        SerializationHelper.write(modelFile, classifier);

        System.out.println("Evaluating model with 3-fold CV...\n");
        Evaluation eval = new Evaluation(train);
        eval.crossValidateModel(classifier, train, 3, new Random(1));
        System.out.println(eval.toSummaryString("\nResults\n======\n", false));

        System.out.println("Creating result file for test instances...\n");
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

        System.out.println("====================================================");
    }

    static Instances ReadInstances(String path) {
        try {
            DataSource source = new DataSource(path);
            Instances instances = source.getDataSet();
            instances.setClassIndex(instances.numAttributes() - 1);
            return instances;
        } catch (Exception e) {
            e.printStackTrace();
        }

        return null;
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
