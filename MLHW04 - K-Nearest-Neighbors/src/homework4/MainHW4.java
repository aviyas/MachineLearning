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
	
	public static void main (String [] args) throws Exception {

		// 1. Finds the combination of k, p and vote function that gives the lowest cross validation error.
		// Checks k = 1,...,30 , p = 1, 2, 3, infinity and both getClassVoteResult() and getWeightedClassVoteResult()

		// 2. Prints the cross validation error with the best parameters for the both glass and cancer data.

		// 3. Compares the three Edited-Knn algorithms,
		// by calculating their error and measure the average elapsed time it takes to do so, excluding training time.

		System.nanoTime();

	}

}
