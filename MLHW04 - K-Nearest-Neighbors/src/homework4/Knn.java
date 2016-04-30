package homework4;

import com.sun.tools.doclint.HtmlTag;
import weka.classifiers.Classifier;
import weka.core.InstanceComparator;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;


import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;


public class Knn extends Classifier {

	private String M_MODE = "";
	Instances m_trainingInstances;

	int k;
	int p;
	String votingMethod = "";

	public String getM_MODE() {
		return M_MODE;
	}

	public void setM_MODE(String m_MODE) {
		M_MODE = m_MODE;
	}

	public String getVotingMethod() { return votingMethod; }

	public void setVotingMethod(String votingMethod) { this.votingMethod = votingMethod; }

	@Override
	public void buildClassifier(Instances arg0) throws Exception {

		m_trainingInstances.setClassIndex(m_trainingInstances.numAttributes() - 1);

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
	 * @param instances - the training data.
     */
	private void editedForward(Instances instances) {
	}

	/**
	 * Trains the algorithm using the Edited-Knn Backwards Algorithm shown in class.
	 * @param instances - the training data.
	 */
	private void editedBackward(Instances instances) {
	}

	/**
	 * Trains the algorithm using the Standard Knn Algorithm shown in class, storing all of the instances in memory.
	 * @param instances - the training data.
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

		if (votingMethod.equals("weighted")) {
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

		HashMap<Double, Instance> allData = new HashMap<Double, Instance>();

		// 1. Calculates the distance from all instances and put in a HashMap
		Instance currentInstance;
		for (int i = 0; i < m_trainingInstances.numInstances(); i++) {
			currentInstance = m_trainingInstances.instance(i);
			allData.put(distance(currentInstance, newInstance), currentInstance);
		}

		// 2. Finds k mappings with k lowest key values
		HashMap<Double, Instance> nearestNeighbors = new HashMap<Double, Instance>();
		Double currentMinValue = 0.0;
		Instance currentMinInstance;

		for (int j = 0; j < k; j++) {
			currentMinValue = Collections.min(nearestNeighbors.keySet());
			currentMinInstance = nearestNeighbors.get(currentMinValue);
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

		double distance = 0;

		// Calculates distance according with p value and attribute type
		if (thingOne.attribute(0).isNumeric()) {
			distance = (Double.isFinite(p)) ? lPDistance(thingOne, thingTwo) : lInfinityDistance(thingOne, thingTwo);
		} else {
			distance = valueDifferenceMeasureDistance(thingOne, thingTwo);
		}

		return distance;
	}

	/**
	 * Calculates the distance between two instances with non-numeric attributes.
	 * @param thingOne first instance
	 * @param thingTwo second instance
     * @return the difference measure distance between the instances.
     */
	private double valueDifferenceMeasureDistance(Instance thingOne, Instance thingTwo) {

		double distance = 0;
		Attribute classAttribute = thingOne.classAttribute();

		for (int i = 0; i < classAttribute.numValues(); i++) {
			distance += (aposterioriProbability(classAttribute.value(i), thingOne) -
							aposterioriProbability(classAttribute.value(i), thingTwo));
		}
		return distance;
	}

	/**
	 * Calculates the a posteriori probability of a given class value if instance is thingOne, using the bayes formula.
	 * @param classValue the given value.
	 * @param thingOne the given instance.
     * @return probability (A = classValue | x = thingOne).
     */
	private double aposterioriProbability(String classValue, Instance thingOne) {
		double classPriorProbability = 0;
		double instancePriorProbability = 0;
		double likelihood = 0;

		// Estimates P(Ai) based on the training set

		// Estimates P(xj) based on the training set

		// Estimates P(xj | Ai) = p(xj and Ai) / P(Ai) based on the training set

		return ((likelihood * classPriorProbability) / instancePriorProbability);
	}

	/**
	 * Calculates the l-p distance between two instances.
	 * @param thingOne first instance
	 * @param thingTwo second instance
	 * @return the l-p distance between the instances.
	 */
	public double lPDistance(Instance thingOne, Instance thingTwo) {

		double distance = 0;

		for (int i = 0; i < thingOne.numAttributes() - 1; i++) {
			distance += Math.pow(thingOne.value(i) - thingTwo.value(i), p);
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

		return -1.0;
	}

	/**
	 * Calculates the average error on given dataset.
	 * @param dataset of instances.
	 * @return the error on the dataset.
     */
	public Double calcAverageError(Instances dataset) {

		return -1.0;
	}

	/**
	 * Calculates the Cross Validation Error, using 10 folds.
	 * @param dataset of instances.
	 * @return the cross validation error on the dataset.
     */
	public Double crossValidationError(Instance dataset) {

		return -1.0;
	}


}
