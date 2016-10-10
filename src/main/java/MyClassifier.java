import weka.classifiers.Classifier;

public class MyClassifier {
    public final Classifier classifier;
    public final String Id;

    public MyClassifier(Classifier classifier, String id) {
        this.classifier = classifier;
        Id = id;
    }
}
