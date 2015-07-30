package in.ac.iitb.cse.qh.data;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class TargetState {
	private int trainingSize;
	private TargetStateVector[] s;

	public int getTrainingSize() {
		return trainingSize;
	}

	public void setTrainingSize(int trainingSize) {
		this.trainingSize = trainingSize;
	}

	public TargetStateVector[] getS() {
		return s;
	}

	public void setS(TargetStateVector[] s) {
		this.s = s;
	}

	public void serialize(String file) {
		int numU = 0;
		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(file));
			for (TargetStateVector vec : getS()) {
				if (vec instanceof TargetStateUVector)
					numU++;
				for (double s : vec.getSi())
					writer.append(s + " ");
				writer.newLine();
			}
			writer.append("Number of target U vectors : " + numU);
			writer.newLine();
			writer.close();
		} catch (IOException e) {

		}
	}

}
