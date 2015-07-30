package in.ac.iitb.cse.qh.data;

import in.ac.iitb.cse.qh.util.MetaConstants;

public class CurrentStateVector {
	private int numLabels;
	private double[] pi;
	private double scalingBeta;
	private double sumExpYi;
	private double sumYi;

	public double getSumExpYi() {
		return sumExpYi;
	}

	public void setSumExpYi(double sumExpYi) {
		this.sumExpYi = sumExpYi;
	}

	public CurrentStateVector() {
		this(MetaConstants.NUMBER_CLASSLABELS, MetaConstants.SCALING_BETA);
	}

	public CurrentStateVector(int numLabels, double beta) {
		this.numLabels = numLabels;
		pi = new double[numLabels];
		scalingBeta = beta;
		sumExpYi = 0.0d;
		sumYi = 0.0d;
	}

	public double[] getPi() {
		return pi;
	}

	public void setPi(double[] yi) {
		// System.out.println("Yi size : " + yi.length);
		for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
			// pi[labelIndex] = Math.exp(scalingBeta * yi[labelIndex]);
			pi[labelIndex] = yi[labelIndex];
//			sumYi += pi[labelIndex];
		}
		// for (int i = 0; i < numLabels; i++) {
		// pi[i] = Math.exp(scalingBeta * (1 - pi[i] / sumYi));
		// sumExpYi += pi[i];
		// }
		// for (int i = 0; i < numLabels; i++)
		// pi[i] /= sumExpYi;
	}

	public int getNumLabels() {
		return numLabels;
	}

	public void setNumLabels(int numLabels) {
		this.numLabels = numLabels;
	}

}
