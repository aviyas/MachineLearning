package homework1;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.Stream;

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

        // Load data
        MainHW1 homework = new MainHW1();
        Instances training = homework.loadData("housing_training.txt");
        Instances testing = homework.loadData("housing_training.txt");

        // Train classifier
        LinearRegression regressionFunc = new LinearRegression();
        regressionFunc.buildClassifier(training);

        Stream<String> lines = Stream.of("The weights for question 7 are:");

        // Write weights into "hw01.txt"
        Stream<String> coeffs = Arrays.stream(regressionFunc.getCoefficients()).skip(1).mapToObj(Double::toString);
        lines = Stream.concat(lines, coeffs);

        // Calculate error and display in "hw01.txt"
        lines = Stream.concat(lines,
                Stream.of(String.format("The error for question 7 is %f", regressionFunc.calculateSE(testing))));

        try {
            Files.write(Paths.get("hw1.txt"), lines.collect(Collectors.toList()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
