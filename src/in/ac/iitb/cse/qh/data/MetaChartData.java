package in.ac.iitb.cse.qh.data;

public class MetaChartData {
	private int iteration;
	private double divergence;

	public MetaChartData(int iter, double div) {
		iteration = iter;
		divergence = div;
	}

	public double getDivergence() {
		return divergence;
	}

	public void setDivergence(double divergence) {
		this.divergence = divergence;
	}

	public int getIteration() {
		return iteration;
	}

	public void setIteration(int iteration) {
		this.iteration = iteration;
	}
}
