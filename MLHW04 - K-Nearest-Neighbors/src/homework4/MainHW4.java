package homework4;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class MainHW4 {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
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


	public static void main (String [] args) throws Exception {

		// Load data
		MainHW4 homework = new MainHW4();
		Instances glassData = homework.loadData("glass.txt");
		Instances cancerData = homework.loadData("cancer.txt");

		// 2. Prints the cross validation error with the best parameters for the both glass and cancer data.

		double[] bestParams = homework.findBestParameters(glassData);

		System.out.println("Best k found: " + bestParams[0]);
		System.out.println("Best p found: " + bestParams[1]);
		System.out.println("Best votingMethod found: " + bestParams[2]);
		System.out.println("Lowest crossValidationError foudn: " + bestParams[3]);

		// 3. Compares the three Edited-Knn algorithms,
		// by calculating their error and measure the average elapsed time it takes to do so, excluding training time.

		System.nanoTime();

	}

	/**
	 * Checks k = 1,...,30 , p = 1, 2, 3, infinity and both getClassVoteResult() - 0 and getWeightedClassVoteResult() - 1,
	 * to find the parameters that give the lowest cross validation error and prints them.
	 * @param data
	 * @throws Exception
	 * @return lowest cross validation error found.
     */
	private double[] findBestParameters(Instances data) throws Exception {

		double crossValidationError = 0;
		double[] bestParams = new double[3];

		Knn knn = new Knn();

		for (int k = 1; k <= 30; k++) {
			for (double p = 1; p <= 4; p++) {
				for (double votingMethod = 0; votingMethod <= 1; votingMethod++) {

					// Goes through p = 1, 2, 3 and infinity
					p = (p == 4) ? Double.POSITIVE_INFINITY : p;

					// Calculates best k and p using votingMethod
					knn.setParameters(k, p, votingMethod);
					knn.buildClassifier(data);
					crossValidationError = knn.crossValidationError(data);

					if (crossValidationError > bestParams[4]) {
						bestParams[4] = crossValidationError;
						bestParams[0] = k;
						bestParams[1] = p;
						bestParams[2] = votingMethod;
					}
				}
			}
		}

		return bestParams;
	}

}
