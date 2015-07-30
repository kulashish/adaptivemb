package in.ac.iitb.cse.qh.data;

import java.util.Arrays;

public class TargetStateVector {
	protected int numLabels;
	protected double[] si;

	protected TargetStateVector(int labels) {
		numLabels = labels;
		si = new double[numLabels];
		Arrays.fill(si, 0.0d);
	}

	public TargetStateVector(CurrentStateVector vector) {
		numLabels = vector.getNumLabels();
		double pi[] = vector.getPi();
		si = new double[numLabels];
		for (int i = 0; i < numLabels; i++)
			si[i] = pi[i];
	}

	public double[] getSi() {
		return si;
	}

	public void setSi(double[] si) {
		this.si = si;
	}

}
