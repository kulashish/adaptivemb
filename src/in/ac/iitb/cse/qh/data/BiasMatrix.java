package in.ac.iitb.cse.qh.data;

import in.ac.iitb.cse.qh.util.MetaConstants;

public class BiasMatrix {
	private int numLabels;
	private int[][] matrix;
	private int tp = 0;
	private int tn = 0;
	private int fp = 0;
	private int fn = 0;

	public BiasMatrix() {
		this(MetaConstants.NUMBER_CLASSLABELS);
	}

	public BiasMatrix(int labels) {
		numLabels = labels;
		matrix = new int[numLabels][numLabels];
	}

	public int getTp() {
		return tp;
	}

	public void setTp(int tp) {
		this.tp = tp;
		matrix[1][1] = tp;
	}

	public int getTn() {
		return tn;
	}

	public void setTn(int tn) {
		this.tn = tn;
		matrix[0][0] = tn;
	}

	public int getFp() {
		return fp;
	}

	public void setFp(int fp) {
		this.fp = fp;
		matrix[0][1] = fp;
	}

	public int getFn() {
		return fn;
	}

	public void setFn(int fn) {
		this.fn = fn;
		matrix[1][0] = fn;
	}

	public int getNumLabels() {
		return numLabels;
	}

	public void setNumLabels(int numLabels) {
		this.numLabels = numLabels;
	}

	public int[][] getMatrix() {
		return matrix;
	}

	public void setMatrix(int[][] matrix) {
		this.matrix = matrix;
		tp = matrix[1][1];
		fp = matrix[0][1];
		tn = matrix[0][0];
		fn = matrix[1][0];
	}

}
