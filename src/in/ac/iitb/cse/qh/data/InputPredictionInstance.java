package in.ac.iitb.cse.qh.data;

public class InputPredictionInstance {
	private int trueLabel;
	private int predLabel;
	private double[] yi;

	public int getTrueLabel() {
		return trueLabel;
	}

	public void setTrueLabel(int trueLabel) {
		this.trueLabel = trueLabel;
	}

	public int getPredLabel() {
		return predLabel;
	}

	public void setPredLabel(int predLabel) {
		this.predLabel = predLabel;
	}

	public double[] getYi() {
		return yi;
	}

	public void setYi(double[] yi) {
		this.yi = yi;
	}

	public InputPredictionInstance(int tlabel, int plabel, double[] scores) {
//		if(tlabel==0||tlabel==2) System.out.println(tlabel);
		trueLabel = tlabel;
		predLabel = plabel;
		yi = scores;
	}
}
