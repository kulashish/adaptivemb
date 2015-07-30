package in.ac.iitb.cse.qh.data;

import java.util.logging.Level;
import java.util.logging.Logger;

import in.ac.iitb.cse.qh.util.MetaConstants;

public class ConfusionMatrix {
	private int numLables;
	private int[][] matrix;
	private int tp = 0;
	private int fp = 0;
	private int fn = 0;
	private int tn = 0;
	private double advPos;
	private double advNeg;

	private static final Logger LOGGER = Logger.getLogger(ConfusionMatrix.class
			.getName());

	static {
		LOGGER.setLevel(Level.INFO);
	}

	public ConfusionMatrix() {
		this(MetaConstants.NUMBER_CLASSLABELS);
	}

	public ConfusionMatrix(int labels) {
		numLables = labels;
		matrix = new int[numLables][numLables];
	}

	public ConfusionMatrix(int[][] is) {
		setMatrix(is);
	}

	public int[][] getMatrix() {
		return matrix;
	}

	public double getAdvPos() {
		return advPos;
	}

	public double getAdvNeg() {
		return advNeg;
	}

	public double getMinAdv() {
		if (advPos > 0 && advNeg > 0)
			return Math.min(advPos, advNeg);
		else if (advPos == 0)
			return advNeg;
		else
			return advPos;
	}

	public void setMatrix(int[][] matrix) {
		this.matrix = matrix;
		tp = matrix[1][1];
		fp = matrix[0][1];
		tn = matrix[0][0];
		fn = matrix[1][0];
		advNeg = tn * 1.0 / (tn + fp) - 0.5;
		advPos = tp * 1.0 / (tp + fn) - 0.5;
	}

	public int getNumLables() {
		return numLables;
	}

	public void setNumLables(int numLables) {
		this.numLables = numLables;
	}

	public void display() {
		// for (int i = 0; i < numLables; i++)
		// for (int j = 0; j < numLables; j++)
		// System.out.print(matrix[i][j] + " ");
		// System.out.println();
		LOGGER.log(Level.INFO, tn + ", " + fp + ", " + fn + ", " + tp);
	}

	public int getTp() {
		return tp;
	}

	public void setTp(int tp) {
		this.tp = tp;
		matrix[1][1] = tp;
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

	public int getTn() {
		return tn;
	}

	public void setTn(int tn) {
		this.tn = tn;
		matrix[0][0] = tn;
	}

	public void addMatrix(ConfusionMatrix mat) {
		setTn(getTn() + mat.getTn());
		setFp(getFp() + mat.getFp());
		setFn(getFn() + mat.getFn());
		setTp(getTp() + mat.getTp());
	}

}
