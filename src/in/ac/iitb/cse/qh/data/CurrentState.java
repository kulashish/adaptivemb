package in.ac.iitb.cse.qh.data;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class CurrentState {

	private int trainingSize;
	private CurrentStateVector[] p;

	public static CurrentState createCurrentState(
			List<InputPredictionInstance> instances) {
		CurrentState cstate = new CurrentState(instances.size());
		CurrentStateVector[] cstatevectors = cstate.getP();
		int index = 0;
//		System.out.println("Setting current state");
		InputPredictionInstance instance = null;
		for (int instIndex = 0; instIndex < instances.size(); instIndex++) {
			instance = instances.get(instIndex);
			cstatevectors[index] = new CurrentStateVector();
			cstatevectors[index++].setPi(instance.getYi());
		}
		return cstate;
	}

	public CurrentState(int size) {
		trainingSize = size;
		p = new CurrentStateVector[size];
	}

	public int getTrainingSize() {
		return trainingSize;
	}

	public void setTrainingSize(int trainingSize) {
		this.trainingSize = trainingSize;
	}

	public CurrentStateVector[] getP() {
		return p;
	}

	public void setP(CurrentStateVector[] p) {
		this.p = p;
	}

	public void serialize(String file) {
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(file));
			for (CurrentStateVector vec : getP()) {
				for (double p : vec.getPi())
					writer.append(p + " ");
				writer.append(vec.getSumExpYi() + " ");
				writer.newLine();
			}
			writer.close();
		} catch (IOException e) {

		}
	}
}
