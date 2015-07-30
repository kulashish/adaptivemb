package in.ac.iitb.cse.qh.classifiers;

import in.ac.iitb.cse.qh.util.MetaConstants;

import java.util.logging.Level;
import java.util.logging.Logger;

import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Optimization;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class ModifiedLogistic extends Logistic {

	private static final long serialVersionUID = 1L;
	private static final Logger LOGGER = Logger
			.getLogger(ModifiedLogistic.class.getName());

	/**
	 * @param args
	 */

	/** The maximum number of iterations. */
	private int m_MaxIts = 2;

	protected double[] d;
	// protected double[] x; // parameters
	private double[] params;
	protected int m_numInstances;

	/** An attribute filter */
	private RemoveUseless m_AttFilter;

	/** The filter used to make attributes numeric. */
	private NominalToBinary m_NominalToBinary;

	/** The filter used to get rid of missing values. */
	private ReplaceMissingValues m_ReplaceMissingValues;

	// private double[][] m_Data;

	private class OptEng extends Optimization {
		private double[] weights;

		/** Class labels of instances */
		private int[] cls;

		/**
		 * Set the weights of instances
		 * 
		 * @param w
		 *            the weights to be set
		 */
		public void setWeights(double[] w) {
			weights = w;
		}

		/**
		 * Set the class labels of instances
		 * 
		 * @param c
		 *            the class labels to be set
		 */
		public void setClassLabels(int[] c) {
			cls = c;
		}

		@Override
		public String getRevision() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		protected double[] evaluateGradient(double[] x) throws Exception {
			// TODO Auto-generated method stub

			// double[] gradient = new double[par.length];
			// for (int i = 0; i < par.length; i++) {
			// for (int j = 0; j < m_numInstances; j++) {
			// gradient[i] += -m_Data[j][i]
			// * (cls[j] - prob(par, m_Data[j], 1));
			// }
			// gradient[i] += d[i] * par[i];
			// }
			//
			// return gradient;

			double[] grad = new double[x.length];
			int dim = m_NumPredictors + 1; // Number of variables per class

			for (int i = 0; i < cls.length; i++) { // ith instance
				double[] num = new double[m_NumClasses - 1]; // numerator of
																// [-log(1+sum(exp))]'
				int index;
				for (int offset = 0; offset < m_NumClasses - 1; offset++) { // Which
																			// part
																			// of
																			// x
					double exp = 0.0;
					index = offset * dim;
					for (int j = 0; j < dim; j++)
						exp += m_Data[i][j] * x[index + j];
					num[offset] = exp;
				}

				double max = num[Utils.maxIndex(num)];
				double denom = Math.exp(-max); // Denominator of
												// [-log(1+sum(exp))]'
				for (int offset = 0; offset < m_NumClasses - 1; offset++) {
					num[offset] = Math.exp(num[offset] - max);
					denom += num[offset];
				}
				Utils.normalize(num, denom);

				// Update denominator of the gradient of -log(Posterior)
				double firstTerm;
				for (int offset = 0; offset < m_NumClasses - 1; offset++) { // Which
																			// part
																			// of
																			// x
					index = offset * dim;
					firstTerm = weights[i] * num[offset];
					for (int q = 0; q < dim; q++) {
						grad[index + q] += firstTerm * m_Data[i][q];
					}
				}

				if (cls[i] != m_NumClasses - 1) { // Not the last class
					for (int p = 0; p < dim; p++) {
						grad[cls[i] * dim + p] -= weights[i] * m_Data[i][p];
					}
				}
			}

			// Ridge: note that intercepts NOT included
			for (int offset = 0; offset < m_NumClasses - 1; offset++) {
				for (int r = 1; r < dim; r++)
					grad[offset * dim + r] += 2 * d[r] * x[offset * dim + r];
			}

			return grad;

		}

		@Override
		protected double objectiveFunction(double[] x) throws Exception {
			// sum w_i^2d_i + lossT(w)

			// double result = 0d;
			// for (int i = 0; i < par.length; i++) {
			// result += par[i] * par[i] * d[i];
			// }
			//
			// // System.out.println("result1=" + result);
			//
			// for (int i = 0; i < m_numInstances; i++) {
			//
			// double p = logprob(par, m_Data[i], cls[i]);
			// result -= p;
			// }
			// // System.out.println("result2=" + result);
			// return result;

			double nll = 0; // -LogLikelihood
			int dim = m_NumPredictors + 1; // Number of variables per class

			for (int i = 0; i < cls.length; i++) { // ith instance

				double[] exp = new double[m_NumClasses - 1];
				int index;
				for (int offset = 0; offset < m_NumClasses - 1; offset++) {
					index = offset * dim;
					for (int j = 0; j < dim; j++)
						exp[offset] += m_Data[i][j] * x[index + j];
				}
				double max = exp[Utils.maxIndex(exp)];
				double denom = Math.exp(-max);
				double num;
				if (cls[i] == m_NumClasses - 1) { // Class of this instance
					num = -max;
				} else {
					num = exp[cls[i]] - max;
				}
				for (int offset = 0; offset < m_NumClasses - 1; offset++) {
					denom += Math.exp(exp[offset] - max);
				}

				nll -= weights[i] * (num - Math.log(denom)); // Weighted NLL
			}

			// Ridge: note that intercepts NOT included
			for (int offset = 0; offset < m_NumClasses - 1; offset++) {
				for (int r = 1; r < dim; r++)
					nll += d[r] * x[offset * dim + r] * x[offset * dim + r];
			}

			return nll;

		}

	}

	public void setNumberofAttributes(int num) {
		m_NumPredictors = num;
	}

	public double[] getWeights() {
		// return x;
		return params;
	}

	public double[][] computeJacobian(int i) {
		double[] x = params;
		double[][] jac = new double[x.length][2];

		double der = 0d;

		for (int j = 0; j < x.length; j++) {
			der += x[j] * m_Data[i][j];
		}

		der = Math.exp(-der);

		der = (-der) / ((1 + der) * (1 + der));
		// System.out.println("DER: " + der);

		for (int j = 0; j < x.length; j++) {
			jac[j][0] = m_Data[i][j] * der;
			jac[j][1] = m_Data[i][j] * (-der);
			// System.out.println(jac[j][0] + ", " + jac[j][1]);
		}

		return jac;
	}

	public double[][] getHessian() {
		double[] x = params;
		double[][] hessian = new double[x.length][x.length];

		LOGGER.log(Level.FINE, "Hessian calculation started...");

		for (int i = 0; i < x.length; i++) {
			for (int j = 0; j < x.length; j++) {
				hessian[i][j] = 0d;
				for (int k = 0; k < m_Data.length; k++) {
					double temp = 0d;
					for (int l = 0; l < m_Data[k].length; l++) {
						temp += x[l] * m_Data[k][l];
					}
					if (temp > MetaConstants.MAX_POWER)
						temp = MetaConstants.MAX_POWER;
					if (temp < -MetaConstants.MAX_POWER)
						temp = -MetaConstants.MAX_POWER;
					temp = Math.exp(-temp);
					temp = (-temp) / ((1 + temp) * (1 + temp));

					// if(temp == 0)
					// System.out.println("\ntemp="+temp);
					hessian[i][j] += m_Data[k][i] * m_Data[k][j] * temp;
				}
				hessian[i][j] = -hessian[i][j];
			}
		}

		LOGGER.log(Level.FINE, "Hessian calculation finished!");

		return hessian;
	}

	public double[][] getIndicator() {
		double[] x = params;
		double[][] B = new double[x.length][x.length];

		for (int i = 0; i < x.length; i++) {
			for (int j = 0; j < x.length; j++) {
				if (i != j)
					B[i][j] = 0;
				else {
					// System.out.println("x: " + x[i] + " d: " + d[j]);
					if (d[j] > MetaConstants.MAX_POWER)
						d[j] = MetaConstants.MAX_POWER;
					if (d[j] < -MetaConstants.MAX_POWER)
						d[j] = -MetaConstants.MAX_POWER;
					B[i][j] = x[i] * Math.exp(d[j]);
				}
			}
		}
		return B;
	}

	public double[] getCovariance() {
		return d;
	}

	private double prob(double[] par, double[] data, double cls) {
		double p = 0.0d;
		for (int i = 0; i < par.length; i++) {
			p += par[i] * data[i];
		}
		if (p > MetaConstants.MAX_POWER)
			p = MetaConstants.MAX_POWER;
		if (p < -MetaConstants.MAX_POWER)
			p = -MetaConstants.MAX_POWER;
		if (cls == 1) {
			if (p >= 0)
				return 1 / (1 + Math.exp(-p));
			else
				return Math.exp(p) / (1 + Math.exp(p));
		} else {
			if (p >= 0)
				return Math.exp(-p) / (1 + Math.exp(-p));
			else
				return 1 / (1 + Math.exp(p));
		}
	}

	private double prob1(double[] par, double[] data, double cls) {
		double p = 0d;
		for (int i = 0; i < par.length; i++) {
			p += par[i] * data[i];
		}
		return p;
	}

	private double logprob(double[] par, double[] data, double cls) {
		double p = 0d;
		for (int i = 0; i < par.length; i++) {
			p += par[i] * data[i];
			// System.out.print("\npar="+par[i]);
			// System.out.print("\ndata="+data[i]);
		}
		// System.out.print("\np1="+p);
		if (p > MetaConstants.MAX_POWER)
			p = MetaConstants.MAX_POWER;
		if (p < -MetaConstants.MAX_POWER)
			p = -MetaConstants.MAX_POWER;
		if (cls == 1) {
			// System.out.print("\np="+p);
			if (p >= 0)
				return -Math.log(1 + Math.exp(-p));
			else
				return (p - Math.log(1 + Math.exp(p)));
		} else {
			// System.out.print("\np="+p);
			if (p >= 0)
				return (-p - Math.log(1 + Math.exp(-p)));
			else
				return -Math.log(1 + Math.exp(p));
		}
	}

	/**
	 * Computes the distribution for a given instance
	 * 
	 * @param instance
	 *            the instance for which distribution is computed
	 * @return the distribution
	 * @throws Exception
	 *             if the distribution can't be computed successfully
	 */
	public double[] distributionForInstance(Instance instance) throws Exception {

		if (instance == null)
			LOGGER.log(Level.WARNING, "instance null");
		// m_ReplaceMissingValues.input(instance);
		// instance = m_ReplaceMissingValues.output();

		// m_AttFilter.input(instance);
		// instance = m_AttFilter.output();

		// m_NominalToBinary.input(instance);
		// instance = m_NominalToBinary.output();

		// Extract the predictor columns into an array
		double[] instDat = new double[m_NumPredictors + 1];
		int j = 1;
		instDat[0] = 1;
		// System.out.println("m_NumPredictors=" + m_NumPredictors);
		for (int k = 0; k <= m_NumPredictors; k++) {
			if (k != m_ClassIndex) {
				instDat[j++] = instance.value(k);
				// System.out.println("instDat[" + (j - 1) + "]=" + instDat[j -
				// 1]
				// + "," + instance.value(k));
			}
		}

		// double[] distribution = new double[2];
		//
		// distribution[0] = prob(x, instDat, 0);
		// distribution[1] = prob(x, instDat, 1);
		// distribution[0]=prob(x, instDat, 0);
		// distribution[1]=prob(x, instDat, 1);
		// double[] distribution = evaluateProbability(instDat);
		double[] distribution = evaluateProbability(instDat);
		return distribution;
	}

	/**
	 * Compute the posterior distribution using optimized parameter values and
	 * the testing instance.
	 * 
	 * @param data
	 *            the testing instance
	 * @return the posterior probability distribution
	 */
	private double[] evaluateProbability(double[] data) {
		double[] prob = new double[m_NumClasses], v = new double[m_NumClasses];

		// Log-posterior before normalizing
		for (int j = 0; j < m_NumClasses - 1; j++) {
			for (int k = 0; k <= m_NumPredictors; k++) {
				v[j] += m_Par[k][j] * data[k];
			}
		}
		v[m_NumClasses - 1] = 0;

		// Do so to avoid scaling problems
		for (int m = 0; m < m_NumClasses; m++) {
			double sum = 0;
			for (int n = 0; n < m_NumClasses - 1; n++)
				sum += Math.exp(v[n] - v[m]);
			prob[m] = 1 / (sum + Math.exp(-v[m]));
			if (prob[m] == 0)
				prob[m] = 1.0e-20;
		}

		return prob;
	}

	public void setHyperparameters(double[] argd) {
		d = argd;
	}

	public double[] getHyperparameters() {
		return d;
	}

	public void setWparameters(double[] argw) {
		params = argw;
	}

	public double[] getWparameters() {
		return params;
	}

	@Override
	public void buildClassifier(Instances train) throws Exception {
		LOGGER.log(Level.FINE, "build classifier...");
		// remove instances with missing class
		train = new Instances(train);
		train.deleteWithMissingClass();

		// Replace missing values
		m_ReplaceMissingValues = new ReplaceMissingValues();
		m_ReplaceMissingValues.setInputFormat(train);
		train = Filter.useFilter(train, m_ReplaceMissingValues);

		// Remove useless attributes
		// m_AttFilter = new RemoveUseless();
		// m_AttFilter.setInputFormat(train);
		// train = Filter.useFilter(train, m_AttFilter);

		// Transform attributes
		m_NominalToBinary = new NominalToBinary();
		m_NominalToBinary.setInputFormat(train);
		train = Filter.useFilter(train, m_NominalToBinary);

		// Save the structure for printing the model
		// m_structure = new Instances(train, 0);

		// Extract data
		m_ClassIndex = train.classIndex();
		m_NumClasses = train.numClasses();
		m_numInstances = train.numInstances();

		int nK = m_NumClasses - 1; // Only K-1 class labels needed
		int nR = m_NumPredictors = train.numAttributes() - 1;
		int nC = train.numInstances();

		m_Data = new double[nC][nR + 1]; // Data values
		int[] Y = new int[nC]; // Class labels
		double[] xMean = new double[nR + 1]; // Attribute means
		double[] xSD = new double[nR + 1]; // Attribute stddev's
		double[] sY = new double[nK + 1]; // Number of classes
		double[] weights = new double[nC]; // Weights of instances
		double totWeights = 0; // Total weights of the instances
		m_Par = new double[nR + 1][nK]; // Optimized parameter values

		for (int i = 0; i < nC; i++) {
			// initialize X[][]
			Instance current = train.instance(i);
			Y[i] = (int) current.classValue(); // Class value starts from 0
			weights[i] = current.weight(); // Dealing with weights
			totWeights += weights[i];

			m_Data[i][0] = 1;
			int j = 1;
			for (int k = 0; k <= nR; k++) {
				if (k != m_ClassIndex) {
					double x = current.value(k);
					m_Data[i][j] = x;
					xMean[j] += weights[i] * x;
					xSD[j] += weights[i] * x * x;
					j++;
				}
			}

			// Class count
			sY[Y[i]]++;
		}

		if ((totWeights <= 1) && (nC > 1))
			throw new Exception(
					"Sum of weights of instances less than 1, please reweight!");

		xMean[0] = 0; // why?
		xSD[0] = 1; // why?
		for (int j = 1; j <= nR; j++) {
			xMean[j] = xMean[j] / totWeights;
			if (totWeights > 1)
				xSD[j] = Math.sqrt(Math.abs(xSD[j] - totWeights * xMean[j]
						* xMean[j])
						/ (totWeights - 1));
			else
				xSD[j] = 0;
		}

		if (m_Debug) {
			// Output stats about input data
			LOGGER.log(Level.FINE, "Descriptives...");
			for (int m = 0; m <= nK; m++)
				LOGGER.log(Level.FINE, sY[m] + " cases have class " + m);
			LOGGER.log(Level.FINE, "\n Variable     Avg       SD    ");
			for (int j = 1; j <= nR; j++)
				LOGGER.log(
						Level.FINE,
						Utils.doubleToString(j, 8, 4)
								+ Utils.doubleToString(xMean[j], 10, 4)
								+ Utils.doubleToString(xSD[j], 10, 4));
		}

		// Normalise input data
		for (int i = 0; i < nC; i++) {
			// System.out.println();
			for (int j = 0; j <= nR; j++) {
				if (xSD[j] != 0) {
					m_Data[i][j] = (m_Data[i][j] - xMean[j]) / xSD[j];
					// System.out.print(m_Data[i][j]+" ");
				}
			}
		}

		if (null == d) {
			LOGGER.log(Level.INFO, "No hyperparameters... assuming default");
			d = new double[nR + 1];
			for (int i = 0; i <= nR; i++)
				// d[i] = 1.0e-8d;
				// d[i] = 10.0d;
				d[i] = 1.0d;
		}
		double x[] = new double[(nR + 1) * nK];
		double[][] b = new double[2][x.length]; // Boundary constraints, N/A
												// here

		// Initialize
		for (int p = 0; p < nK; p++) {
			int offset = p * (nR + 1);
			x[offset] = Math.log(sY[p] + 1.0) - Math.log(sY[nK] + 1.0); // Null
																		// model
			b[0][offset] = Double.NaN;
			b[1][offset] = Double.NaN;
			for (int q = 1; q <= nR; q++) {
				x[offset + q] = 0.0;
				b[0][offset + q] = Double.NaN;
				b[1][offset + q] = Double.NaN;
			}
		}

		// Warm Start
		// if (null != params)
		// for (int q = 1; q <= nR; q++) {
		// x[q] = params[q];
		// }

		OptEng opt = new OptEng();
		opt.setDebug(m_Debug);
		opt.setWeights(weights);
		opt.setClassLabels(Y);

		// System.out.println();

		// int trainIterations=0;
		if (m_MaxIts == -1) { // Search until convergence
			// trainIterations++;
			// System.out.println("\nRunning trainIterations="+trainIterations);
			x = opt.findArgmin(x, b);
			while (x == null) {
				x = opt.getVarbValues();
				if (m_Debug)
					LOGGER.log(Level.WARNING,
							"200 iterations finished, not enough!");
				// trainIterations++;
				// System.out.println("Running trainIterations="+trainIterations);
				x = opt.findArgmin(x, b);
			}
			if (m_Debug)
				LOGGER.log(Level.INFO,
						" -------------<Converged>--------------");
		} else {
			opt.setMaxIteration(m_MaxIts);
			// trainIterations++;
			// System.out.println("Running trainIterations="+trainIterations);
			x = opt.findArgmin(x, b);
			if (x == null) // Not enough, but use the current value
				x = opt.getVarbValues();
		}

		m_LL = -opt.getMinFunction(); // Log-likelihood

		// Don't need data matrix anymore
		// m_Data = null;
		// System.out.println("Printing weights: ");
		// for (int i = 0; i <= nR; i++) {
		// System.out.println(x[i]);
		// }

		/*
		 * double mod2 = 0d; for (int i = 0; i <= nR; i++) { mod2 += x[i] *
		 * x[i]; } for (int i = 0; i <= nR; i++) { x[i] /= Math.sqrt(mod2); }
		 * System.out.println("||w||^2=" + mod2);
		 */
		// Convert coefficients back to non-normalized attribute units
		for (int i = 0; i < nK; i++) {
			m_Par[0][i] = x[i * (nR + 1)];
			for (int j = 1; j <= nR; j++) {
				m_Par[j][i] = x[i * (nR + 1) + j];
				if (xSD[j] != 0) {
					m_Par[j][i] /= xSD[j];
					m_Par[0][i] -= m_Par[j][i] * xMean[j];
				}
			}
		}
		params = new double[m_Par.length];
		for (int i = 0; i < params.length; i++)
			params[i] = m_Par[i][0];
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

	public double[] getData(int index) {
		// System.out.println("data size: " + m_Data[index].length);
		return m_Data[index];
	}

	public void setMaxIts(int it) {
		m_MaxIts = it;
	}

	public int getMaxIts() {
		return m_MaxIts;
	}

	public void setNumClasses(int num) {
		m_NumClasses = num;
	}

}
