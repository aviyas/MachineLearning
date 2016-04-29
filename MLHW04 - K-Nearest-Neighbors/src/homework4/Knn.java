package homework4;

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
	 * Finds the given k nearest neighbors of a given instance, using a K-d Tree structure.
	 * @param newInstance
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
	 * @param nearestNeighbors - a list of k nearest neighbors.
	 * @return the class value with the most votes.
     */
	public double getClassVoteResult(HashMap<Double, Instance> nearestNeighbors) {

		// 1. Creates mappings of possible class values and their count
		HashMap<Double, Integer> counter = new HashMap<Double, Integer>();

		Double currentClassValue;
		Integer currentCount;

		for (Instance currentInstance : nearestNeighbors.values()) {
			currentClassValue = currentInstance.classValue();
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
	 * @param nearestNeighbors
	 * @return the class value with the most votes.
     */
	public double getWeightedClassVoteResult(HashMap<Double, Instance> nearestNeighbors) {

		// Instead of giving one vote to every class, gives a vote of 1 / (distance)^2.

		return -1.0;
	}

	/**
	 * Calculates the distance between two instances, based on the chosen distance function.
	 * @param thingOne first instance
	 * @param thingTwo second instance
     * @return the distance between the instances.
     */
	public Double distance(Instance thingOne, Instance thingTwo) {


		return -1.0;
	}

	/**
	 * Calculates the l-p distance between two instances.
	 * @param thingOne first instance
	 * @param thingTwo second instance
	 * @return the l-p distance between the instances.
	 */
	public Double lPdistance(Instance thingOne, Instance thingTwo) {

		return -1.0;
	}

	/**
	 * Calculates the l-infinity distance between two instances.
	 * @param thingOne first instance
	 * @param thingTwo second instance
	 * @return the l-infinity distance between the instances.
     */
	public Double lInfinityDistance(Instance thingOne, Instance thingTwo) {

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
