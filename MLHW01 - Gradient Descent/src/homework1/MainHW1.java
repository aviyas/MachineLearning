package homework1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class MainHW1 {

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    public Instances generate1DLinData(int numInstances) {
        Attribute Attribute1 = new Attribute("x");
        Attribute ClassAttribute = new Attribute("y");
        FastVector fvWekaAttributes = new FastVector(2);
        fvWekaAttributes.addElement(Attribute1);
        fvWekaAttributes.addElement(ClassAttribute);
        Instances data = new Instances("Rel", fvWekaAttributes, numInstances);
        data.setClassIndex(1);
        // y = 3x
        for (int i = 0; i < numInstances; i++) {
            Instance iExample = new Instance(2);
            int x = ThreadLocalRandom.current().nextInt(0, 100);
            int y = x * 3;
            iExample.setValue((Attribute) fvWekaAttributes.elementAt(0), x);
            iExample.setValue((Attribute) fvWekaAttributes.elementAt(1), y);
            data.add(iExample);
        }

        return data;

    }

    /**
     * Sets the class index as the last attribute.
     * @param fileName
     * @return Instances data
     * @throws IOException
     */
    public Instances loadData(String fileName) throws IOException{
        BufferedReader datafile = readDataFile(fileName);

        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static void main(String[] args) throws Exception {

        //load data
        MainHW1 homework = new MainHW1();
        Instances training = homework.loadData("housing_training.txt");
        Instances testing = homework.loadData("housing_training.txt");

        //train classifier
        LinearRegression regressionFunc = new LinearRegression();
        regressionFunc.buildClassifier(training);

        //calculate error
        //regressionFunc.calculateSE(testing, )

    }

}
