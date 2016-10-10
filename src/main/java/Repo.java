import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.NormalizedPolyKernel;
import weka.classifiers.meta.Stacking;
import weka.classifiers.trees.RandomForest;

public enum Repo {

    Classifiers;

    public final MyClassifier RandomForest;
    public final MyClassifier SvmWithNormalizedPolyKernel;
    public final MyClassifier SvmDefault;
    public final MyClassifier LogisticRegression;
    public final MyClassifier EnsembleMetaLogisticWithSvmNormKernAndRandomForest;
    public final MyClassifier NaiveBayes;


    Repo() {

        SvmDefault = new MyClassifier(new SMO(), "smo");

        NaiveBayes = new MyClassifier(new NaiveBayes(), "nb");

        RandomForest rf = new RandomForest();
        RandomForest = new MyClassifier(rf, "rf");

        SMO smoNormalizedKernel = new SMO();
        smoNormalizedKernel.setKernel(new NormalizedPolyKernel());
        SvmWithNormalizedPolyKernel = new MyClassifier(smoNormalizedKernel, "SvmWithNormalizedPolyKernel");

        LogisticRegression = new MyClassifier(new Logistic(), "LogisticRegression");

        Stacking stacking = new Stacking();
        stacking.setMetaClassifier(new Logistic());
        stacking.setClassifiers(new Classifier[]{rf, smoNormalizedKernel});
        EnsembleMetaLogisticWithSvmNormKernAndRandomForest = new MyClassifier(stacking, "EnsembleMetaLogisticWithSvmNormKernAndRandomForest");


    }
}
