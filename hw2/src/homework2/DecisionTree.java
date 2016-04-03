package homework2;

import weka.classifiers.Classifier;
import weka.core.Instances;

public class DecisionTree extends Classifier{

    private boolean m_pruningMode = false;

    @Override
    public void buildClassifier(Instances arg0) throws Exception {
        //

    }

    public void setPruningMode(boolean pruningMode){
        m_pruningMode = pruningMode;
    }

}
