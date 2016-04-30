package homework4;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.SystemInfo;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.TreeMap;


public class Knn extends Classifier {

	private String M_MODE = "";
	Instances m_trainingInstances;

	double k;
	double p;
	double votingMethod;

	public String getM_MODE() { return M_MODE; }

	public void setM_MODE(String m_MODE) {
		M_MODE = m_MODE;
	}

	public void setParameters(double k, double p, double votingMethod) {
		this.k = k;
		this.p = p;
		this.votingMethod = votingMethod;
	}

	@Override
	public void buildClassifier(Instances arg0) throws Exception {

//		m_trainingInstances.setClassIndex(m_trainingInstances.numAttributes() - 1);

		switch (M_MODE){
		case "none":
			noEdit(arg0);
			break;
		case "forward":
			editedForward(arg0);
			break;
		case "backward":
			editedBackward(arg0);
			break;
		default:
			noEdit(arg0);
			break;
		}
	}

	/**
	 * Trains the algorithm using the Edited-Knn Forwards Algorithm shown in class.
	 * @param instances of training data.
     */
	private void editedForward(Instances instances) {
	}

	/**
	 * Trains the algorithm using the Edited-Knn Backwards Algorithm shown in class.
	 * @param instances of training data.
	 */
	private void editedBackward(Instances instances) {
	}

	/**
	 * Trains the algorithm using the Standard Knn Algorithm shown in class, storing all of the instances in memory.
	 * @param instances of training data.
     */
	private void noEdit(Instances instances) {
		m_trainingInstances = new Instances(instances);
	}

	/**
	 * Predicts the class value of a given instance.
	 * @param newInstance to classify.
	 * @return the classification.
     */
	public double classify(Instance newInstance) {

		HashMap<Double, Instance> nearestNeighbors = findNearestNeighbors(newInstance);

		if (votingMethod == 1) {
			return getWeightedClassVoteResult(nearestNeighbors);
		} else {
			return getClassVoteResult(nearestNeighbors);
		}
	}

	/**
	 * Finds the given k nearest neighbors of a given instance, using the standard method.
	 * @param newInstance to be examined.
	 * @return k nearest neighbors and their distances.
     */
	public HashMap<Double, Instance> findNearestNeighbors(Instance newInstance) {

		TreeMap<Double, Instance> allData = new TreeMap<>();

		// 1. Calculates the distance from all instances and put in a TreeMap
		Instance currentInstance;
		for (int i = 0; i < m_trainingInstances.numInstances(); i++) {
			currentInstance = m_trainingInstances.instance(i);
			allData.put(distance(currentInstance, newInstance), currentInstance);
		}

		if (allData.size() < k) {
			System.out.println("problem with finding nearestNeighbors, params: k = " + k + " , p = " + p);
		}

		// 2. Finds k mappings with k lowest key values
		HashMap<Double, Instance> nearestNeighbors = new HashMap<>();
		Double currentMinValue = 0.0;
		Instance currentMinInstance;

		for (int j = 0; j < k; j++) {

			currentMinValue = allData.firstEntry().getKey();
			currentMinInstance = allData.firstEntry().getValue();
			nearestNeighbors.put(currentMinValue, currentMinInstance);
			allData.remove(currentMinValue, currentMinInstance);
		}

		return nearestNeighbors;
	}

	/**
	 * Takes a vote on what the class of the neighbors are and determines the final result accordingly.
	 * @param nearestNeighbors to base the vote on.
	 * @return the class value with the most votes.
     */
	public double getClassVoteResult(HashMap<Double, Instance> nearestNeighbors) {

		// 1. Creates mappings of possible class values and their count
		HashMap<Double, Integer> counter = new HashMap<Double, Integer>();

		Double currentClassValue;
		Integer currentCount;

		for (Instance neighbor : nearestNeighbors.values()) {
			currentClassValue = neighbor.classValue();
			currentCount = counter.getOrDefault(currentClassValue, 0);
			counter.put(currentClassValue, currentCount + 1);
		}

		int maxCount = 0;
		double vote = -1;

		// 2. Finds maximum among mappings
		for (Double classValue : counter.keySet()) {
			currentCount = counter.get(classValue);
			if (currentCount > maxCount) {
				maxCount = currentCount;
				vote = classValue;
			}
		}
		return vote;
	}

	/**
	 * Takes a vote on what the class of the neighbors are, weighted according with their distances from the newInstance,
	 * and determines the final result accordingly.
	 * @param nearestNeighbors to base the vote on.
	 * @return the class value with the most votes.
     */
	public double getWeightedClassVoteResult(HashMap<Double, Instance> nearestNeighbors) {


		// 1. Creates mappings of possible class values and their rating
		HashMap<Double, Double> rater = new HashMap<Double, Double>();

		Instance currentNeighbor;
		Double currentClassValue;
		Double currentRate;
		Double addedRate;

		for (Double distance : nearestNeighbors.keySet()) {
			currentNeighbor = nearestNeighbors.get(distance);
			currentClassValue = currentNeighbor.classValue();
			currentRate = rater.getOrDefault(currentClassValue, 0.0);

			// Instead of giving one vote to every class, gives a vote of 1 / (distance)^2.
			addedRate = 1 / Math.pow(distance, 2);
			rater.put(currentClassValue, currentRate + addedRate);
		}

		double maxRate = 0;
		double vote = -1;

		// 2. Finds maximum among mappings
		for (Double classValue : rater.keySet()) {
			currentRate = rater.get(classValue);
			if (currentRate > maxRate) {
				maxRate = currentRate;
				vote = classValue;
			}
		}
		return vote;
	}

	/**
	 * Calculates the distance between two instances, based on the chosen distance function.
	 * @param thingOne first instance
	 * @param thingTwo second instance
     * @return the distance between the instances.
     */
	public double distance(Instance thingOne, Instance thingTwo) {
		return (p == Double.MAX_VALUE) ? lInfinityDistance(thingOne, thingTwo) : lPDistance(thingOne, thingTwo);
	}

	/**
	 * Calculates the l-p distance between two instances.
	 * @param thingOne first instance
	 * @param thingTwo second instance
	 * @return the l-p distance between the instances.
	 */
	public double lPDistance(Instance thingOne, Instance thingTwo) {

		double distance = 0;

		// Sums all differences at the power of p
		for (int i = 0; i < thingOne.numAttributes() - 1; i++) {
			distance += Math.abs(Math.pow(thingOne.value(i) - thingTwo.value(i), p));
		}
		return root(distance, p);
	}

	/**
	 * Calculates the nth root of a number.
	 * @param number the value under the root.
	 * @param n the degree of the calculation.
     * @return the nth root of value.
     */
	private double root(double number, double n) {
		return Math.pow(Math.E, Math.log(number) / n);
	}

	/**
	 * Calculates the l-infinity distance between two instances.
	 * @param thingOne first instance
	 * @param thingTwo second instance
	 * @return the l-infinity distance between the instances.
     */
	public double lInfinityDistance(Instance thingOne, Instance thingTwo) {

		double maxDistance = Integer.MIN_VALUE;
		double currentDistance;

		// Takes the maximum difference measured between all attributes
		for (int i = 0; i < thingOne.numAttributes() - 1; i++) {
			currentDistance = Math.abs(thingOne.value(i) - thingTwo.value(i));
			maxDistance = (currentDistance > maxDistance) ? currentDistance : maxDistance;
		}
		return maxDistance;
	}

	/**
	 * Calculates the average error on given dataset: # mistakes / # instances.
	 * @param dataset of instances.
	 * @return the error on the dataset.
     */
	public double calcAverageError(Instances dataset) {

		int mistakes = 0;

		for (int i = 0; i < dataset.numInstances(); i++) {
			mistakes = (classify(dataset.instance(i)) != dataset.instance(i).classValue()) ? mistakes + 1 : mistakes;
		}
		return (mistakes / dataset.numInstances());
	}

	/**
	 * Calculates the Cross Validation Error, using 10 folds.
	 * @param dataset of instances.
	 * @return the cross validation error on the dataset.
     */
	public double crossValidationError(Instances dataset) throws Exception {

		double error = 0;
		Instances[] splitData;

		for (int i = 0; i < 10; i++) {
			splitData = splitDataBy(dataset, i);
			buildClassifier(splitData[0]);
			error += calcAverageError(splitData[1]);
		}
		return error / 10;
	}

	/**
	 * Splits the dataset into 10 subsets by index i.
	 * @param dataset to split.
	 * @param index of fold to set as validation.
	 * @return an array where arr[0] = 9 learning folds and arr[1] = 1 validation fold.
     */
	private Instances[] splitDataBy(Instances dataset, int index) {

		Instances[] splitData = new Instances[2];

		int foldSize = (int)(dataset.numInstances() / 10);
		int splitIndex = foldSize * index;

		// 1. Copies the relevant validation section
		splitData[1] = new Instances(dataset, splitIndex, foldSize);

		// 2. Copies the relevant training sections:
		// 		2.1. Pre validation set
		splitData[0] = new Instances(dataset, 0, splitIndex);

		// 		2.2. Post validation set
		for (int i = splitIndex + foldSize; i < dataset.numInstances(); i++) {
				splitData[0].add(dataset.instance(i));
		}
		return splitData;
	}

}