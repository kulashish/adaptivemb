package in.ac.iitb.cse.qh.meta;

import in.ac.iitb.cse.qh.data.CurrentState;
import in.ac.iitb.cse.qh.data.CurrentStateVector;
import in.ac.iitb.cse.qh.data.InputData;
import in.ac.iitb.cse.qh.data.InputPredictionInstance;
import in.ac.iitb.cse.qh.data.TargetState;
import in.ac.iitb.cse.qh.data.TargetStateIVector;
import in.ac.iitb.cse.qh.data.TargetStateUVector;
import in.ac.iitb.cse.qh.data.TargetStateVector;

public class TargetStateCalculator {
	private TargetState targetState;
	private InputPredictionInstance[] inputInstances;
	private int[][] bias;
	private int[][] conf;
	private int size;
	private int numLabels;
	private CurrentStateVector[] cStateVectors;

	public TargetStateCalculator(InputData data, CurrentState currentState) throws Exception {
		size = data.getPredInstances().size();
		inputInstances = (InputPredictionInstance[]) data.getPredInstances()
				.toArray(new InputPredictionInstance[size]);
		bias = data.getBiasMatrix().getMatrix();
		conf = data.getConfMatrix().getMatrix();
		cStateVectors = currentState.getP();
	}

	public TargetState calculate() {
		targetState = new TargetState();
		targetState.setTrainingSize(size);
		TargetStateVector[] targetvectors = new TargetStateVector[size];
		targetState.setS(targetvectors);
		InputPredictionInstance inst = null;
		int tLabel = -1;
		int pLabel = -1;
		for (int instanceIndex = 0; instanceIndex < size; instanceIndex++) {
			inst = inputInstances[instanceIndex];
			tLabel = inst.getTrueLabel();
			pLabel = inst.getPredLabel();
			if (bias[tLabel][pLabel] < conf[tLabel][pLabel]) { // Bias
																// Down
				targetvectors[instanceIndex] = new TargetStateUVector(pLabel);
			} else {
				boolean blnChange = false;
				for (int label = 0; label < numLabels; label++)
					if (label != pLabel
							&& bias[tLabel][label] > conf[tLabel][label]) { // Bias
																			// Up
						blnChange = true;
						targetvectors[instanceIndex] = new TargetStateIVector(
								numLabels, pLabel);
					}
				if (!blnChange)
					targetvectors[instanceIndex] = new TargetStateVector(
							cStateVectors[instanceIndex]);
			}
		}
		return targetState;
	}
}
