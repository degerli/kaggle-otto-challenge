import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.NormalizedPolyKernel;
import weka.classifiers.meta.Stacking;
import weka.classifiers.trees.RandomForest;

public enum Repo {

    Classifiers;

    public final MyClassifier RANDOM_FOREST;
    public final MyClassifier SMO_NORMALIZED_KERNEL;
    public final MyClassifier LOGISTIC;
    public final MyClassifier LOG_RF_SMO;


    Repo() {

        RandomForest rf = new RandomForest();
        RANDOM_FOREST = new MyClassifier(rf, "rf");

        SMO smo = new SMO();
        smo.setKernel(new NormalizedPolyKernel());
        SMO_NORMALIZED_KERNEL = new MyClassifier(smo, "smo_norm_kern");

        LOGISTIC = new MyClassifier(new Logistic(), "logit");

        Stacking stacking = new Stacking();
        stacking.setMetaClassifier(new Logistic());
        stacking.setClassifiers(new Classifier[]{rf, smo});
        LOG_RF_SMO = new MyClassifier(stacking, "logit-rf-smo");


    }
}
