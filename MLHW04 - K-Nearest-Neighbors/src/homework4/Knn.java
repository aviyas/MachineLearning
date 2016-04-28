package homework4;

import weka.classifiers.Classifier;
import weka.core.Instances;


public class Knn extends Classifier {

	private String M_MODE = "";
	Instances m_trainingInstances;

	public String getM_MODE() {
		return M_MODE;
	}

	public void setM_MODE(String m_MODE) {
		M_MODE = m_MODE;
	}

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
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
	
	private void editedForward(Instances instances) {
	}
	
	private void editedBackward(Instances instances) {
	}
	
	private void noEdit(Instances instances) {
		m_trainingInstances = new Instances(instances);
	}

}
