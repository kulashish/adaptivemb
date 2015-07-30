package in.ac.iitb.cse.qh.meta;

import in.ac.iitb.cse.qh.classifiers.ModifiedLogistic;
import in.ac.iitb.cse.qh.data.ConfusionMatrix;
import in.ac.iitb.cse.qh.data.CurrentState;
import in.ac.iitb.cse.qh.data.CurrentStateVector;
import in.ac.iitb.cse.qh.data.InputData;
import in.ac.iitb.cse.qh.data.InputPredictionInstance;
import in.ac.iitb.cse.qh.data.MetaChartBean;
import in.ac.iitb.cse.qh.data.ModelParams;
import in.ac.iitb.cse.qh.data.TargetState;
import in.ac.iitb.cse.qh.data.TargetStateVector;
import in.ac.iitb.cse.qh.util.BeanFinder;
import in.ac.iitb.cse.qh.util.KLDivergenceCalculator;
import in.ac.iitb.cse.qh.util.MetaConstants;
import in.ac.iitb.cse.qh.util.WekaUtil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.ConsoleHandler;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Optimization;
import Jama.Matrix;

public class Optimizer {
	protected InputData data;
	private CurrentState cState;
	private TargetState tState;
	protected int numLabels;
	private double divergence;
	protected InputData newData;
	private int numWeights;
	// private HyperparameterLearner hyperLearner;
	private ClassifierProxy classifier;

	protected boolean m_Debug;

	private int m_MaxIts = -1;

	private final static Logger LOGGER = Logger.getLogger(Optimizer.class
			.getName());

	static {
		LOGGER.setLevel(Level.INFO);
		Handler consoleHandler = new ConsoleHandler();
		consoleHandler.setLevel(Level.FINER);
		LOGGER.addHandler(consoleHandler);
	}

	private class OptEng extends Optimization {

		@Override
		public String getRevision() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		protected double[] evaluateGradient(double[] x) throws Exception {
			LOGGER.log(Level.FINEST, "Computing Gradient...");
			double[] grad = computeGradient(x);
			LOGGER.log(Level.FINEST, "Gradient Computation Finished");
			return grad;

		}

		@Override
		protected double objectiveFunction(double[] x) throws Exception {
			getNewState(x);
			double kldiv = KLDivergenceCalculator.calculate(cState, tState);
			LOGGER.log(Level.FINEST, "kldiv=" + kldiv);
			return kldiv;
		}
	}

	public Optimizer(InputData in, CurrentState curr, TargetState target,
			ClassifierProxy classifier) {
		LOGGER.log(Level.INFO, "Log level : " + LOGGER.getLevel());
		this.classifier = null != classifier ? classifier
				: new ClassifierProxy();
		data = in;
		cState = curr;
		tState = target;
		numLabels = MetaConstants.NUMBER_CLASSLABELS;
		numWeights = data.getParams().getParams().length;
		// hyperLearner = new HyperparameterLearner(classifier, data.getParams()
		// .getParams());
	}

	public double[] computeGradient(double[] theta) throws Exception {
		int numInstances = cState.getTrainingSize();
		CurrentStateVector[] cStateVectors = cState.getP();
		TargetStateVector[] tStateVectors = tState.getS();
		double[][] jacobian = null;
		Matrix jacMatrix = null;

		Matrix gradMatrix = new Matrix(numWeights, 1); // D x 1
		double[] grads = new double[numWeights];
		Arrays.fill(grads, 0.0d);
		Matrix dataMat = null;
		double temp = 0.0d;
		HyperparameterLearner hyperLearner = new HyperparameterLearner(
				classifier, theta);

		Instances holdoutInstances = classifier.getholdoutInstances();
		int i = 0;
		for (Instance instance : holdoutInstances) {
			double[] instDat = new double[instance.numAttributes()];
			int j = 1;
			instDat[0] = 1;
			for (int k = 0; k < instance.numAttributes() - 1; k++) {
				instDat[j++] = instance.value(k);
			}

			dataMat = new Matrix(instDat, 1);

			// //for (int i = 0; i < numInstances; i++) {
			// jacobian = classifier.computeJacobian(i); // D x M
			// jacMatrix = new Matrix(jacobian);
			// //dataMat = new Matrix(classifier.getData(i), 1);
			// System.out.println("data matrix dimensions: "
			// + dataMat.getRowDimension() + " x "
			// + dataMat.getColumnDimension());

			temp = tStateVectors[i].getSi()[0] * cStateVectors[i].getPi()[1]
					- tStateVectors[i].getSi()[1] * cStateVectors[i].getPi()[0];
			if (Double.isNaN(temp))
				LOGGER.log(Level.WARNING, "Temp is NAN!!!");

			gradMatrix.plusEquals(dataMat.times(temp).transpose());
			i++;
		}
		for (i = 0; i < numWeights; i++)
			if (Double.isNaN(gradMatrix.get(i, 0)))
				LOGGER.log(Level.WARNING, String.valueOf(gradMatrix.get(i, 0)));
		Matrix hyperJacMat = hyperLearner.computeJacobian(theta);
		// System.out.println("dimensions of hyperparam jacobian: "
		// + hyperJacMat.getRowDimension() + " x "
		// + hyperJacMat.getColumnDimension());
		return hyperJacMat.transpose().times(gradMatrix).getRowPackedCopy();
	}

	public ModelParams optimize2() throws Exception {
		// Initialize
		boolean optim = false;
		double theta[] = data.getParams().getParams();
		double[][] b = new double[2][theta.length]; // Boundary constraints, N/A
		if (LOGGER.getLevel() == Level.FINE)
			LOGGER.log(Level.FINE, "Initial hyperparam: " + theta);
		// for (int i = 0; i < theta.length; i++)
		// System.out.print(theta[i] + ", ");
		// System.out.println();
		MetaChartBean chart = (MetaChartBean) BeanFinder
				.findBean(MetaConstants.BEAN_DIVERGENCE_CHART);
		// if (null != chart)
		// chart.reset();

		for (int p = 0; p < theta.length; p++) {
			// theta[p] = 1.0;
			b[0][p] = 1.0e-8;// -MetaConstants.MAX_POWER;
			b[1][p] = MetaConstants.MAX_POWER;
		}

		OptEng opt = new OptEng();
		opt.setDebug(m_Debug);
		// opt.setWeights(weights);
		// opt.setClassLabels(Y);

		LOGGER.log(Level.FINE, "Initial confusion matrix : ");
		data.getConfMatrix().display();
		int iterCount = 0;
		double minima = Double.MAX_VALUE;
		do {
			m_MaxIts = 1;
			opt.setMaxIteration(m_MaxIts);
			iterCount++;
			if (LOGGER.getLevel() == Level.FINE)
				LOGGER.log(Level.FINE,
						"\nRunning hyperparameterLearning iteration count="
								+ iterCount);
			theta = opt.findArgmin(theta, b);
			if (null == theta)
				theta = opt.getVarbValues();
			if (minima > opt.getMinFunction())
				minima = opt.getMinFunction();
			// else
			// break;
			if (LOGGER.getLevel() == Level.FINE) {
				LOGGER.log(Level.FINE, "\nKL div = " + opt.getMinFunction());
				LOGGER.log(Level.FINE, "\nKL div minima= " + minima);
			}
			// curIterations++;
			// MetaChartBean chart = (MetaChartBean) BeanFinder
			// .findBean(MetaConstants.BEAN_DIVERGENCE_CHART);

			if (null != chart) {
				// chart.addData(opt.getMinFunction());
				chart.addModel(newData);
				chart.addData(newData.getConfMatrix().getFp());
			}
			optim = optimized(theta);
			if (LOGGER.getLevel() == Level.FINE)
				LOGGER.log(Level.FINE, "\noptim = " + optim);
		} while (!optim && iterCount < data.getMaxIterations());

		if (LOGGER.getLevel() == Level.FINE)
			LOGGER.log(Level.FINE, "OPTIMIZED: " + optim);
		LOGGER.log(Level.FINE, "After optimization confusion matrix : ");
		newData.getConfMatrix().display();

		ModelParams optimParams = new ModelParams();
		optimParams.setOptim(optim);
		if (optim){
			LOGGER.log(Level.INFO, "**********************OPTIMIZED!!*********************");
			optimParams.setParams(theta);
		}
		else
			optimParams.setParams(data.getParams().getParams());
		return optimParams;
	}

	// 2nd iteration should always be robust.
	// have lot more examples for testing.
	// training on larger data set.

	public static void gridSearch(String trainFile, String holdoutFile,
			String testFile) throws Exception {

		ModifiedLogistic mLog = new ModifiedLogistic();
		ConfusionMatrix carr[] = new ConfusionMatrix[11];
		ModifiedLogistic marr[] = new ModifiedLogistic[11];
		double accuracy = 0.0;
		int numModels = 0;
		int stepSize = 60;
		double fn = 1000000;
		// Instances train = WekaUtil.getInstances("liver-train-11.arff");
		// Instances holdout = WekaUtil.getInstances("liver-holdout-11.arff");
		// Instances test = WekaUtil.getInstances("liver-test-1.arff");
		Instances train = WekaUtil.getInstances(trainFile);
		Instances holdout = WekaUtil.getInstances(holdoutFile);
		Instances test = WekaUtil.getInstances(testFile);
		mLog.buildClassifier(train);
		// double h[] = mLog.getHyperparameters();
		// for(int i=0;i<h.length;i++)
		// System.out.println(h[i]);

		ConfusionMatrix base = validateModel(mLog, holdout);
		base.display();
		double threshold = 0.1;
		double tp = base.getTp();
		double tn = base.getTn();
		double fp = base.getFp();
		fn = base.getFn();
		// accuracy = ((tp + tn) / (tp + tn + fp + fn)) - threshold;
		accuracy = tn / (tn + fp) - threshold;
		long count = 0;
		System.out.println("Entering Grid Search...");
		for (int h1 = 0; h1 <= 300; h1 += stepSize) {
			for (int h2 = 0; h2 <= 300; h2 += stepSize) {
				for (int h3 = 0; h3 <= 300; h3 += stepSize) {
					for (int h4 = 0; h4 <= 300; h4 += stepSize) {
						for (int h5 = 0; h5 <= 300; h5 += stepSize) {
							for (int h6 = 0; h6 <= 300; h6 += stepSize) {
								for (int h7 = 0; h7 <= 300; h7 += stepSize) {
									count++;
									mLog = new ModifiedLogistic();

									double theta[] = { h1, h2, h3, h4, h5, h6,
											h7 };
									mLog.setHyperparameters(theta);
									mLog.buildClassifier(train);
									ConfusionMatrix cm = validateModel(mLog,
											holdout);
									double tpp, fpp, fnn, tnn;
									tpp = cm.getTp();
									fpp = cm.getFp();
									fnn = cm.getFn();
									tnn = cm.getTn();
									// double acc = ((tpp + tnn) / (tpp + tnn
									// + fpp + fnn));
									double acc = tnn / (tnn + fpp);
									if ((accuracy < acc) && (fnn <= fn)) {
										if (numModels == 10) {
											int n = numModels;
											while (n > 0
													&& carr[n - 1].getFn() > cm
															.getFn()) {
												carr[n] = carr[n - 1];
												marr[n] = marr[n - 1];
												n--;
											}
											carr[n] = cm;
											marr[n] = mLog;
										} else {
											int n = numModels;
											while (n > 0
													&& carr[n - 1].getFn() > cm
															.getFn()) {
												carr[n] = carr[n - 1];
												marr[n] = marr[n - 1];
												n--;
											}
											carr[n] = cm;
											marr[n] = mLog;
											if (numModels != 10)
												numModels++;
										}
									}
									if (count % 100000 == 0)
										System.out.println("Finished " + count
												+ " iterations");
								}
							}
						}
					}
				}
			}
		}

		for (int i = 0; i < numModels; i++)
			carr[i].display();
		System.out.println("Displaying matrix on test data");
		ConfusionMatrix ctest = validateModel(marr[0], test);
		ctest.display();
	}

	private static ConfusionMatrix validateModel(ModifiedLogistic m,
			Instances validationSet) throws Exception {
		List<InputPredictionInstance> predInst = new ArrayList<InputPredictionInstance>();
		ConfusionMatrix confMatrix = new ConfusionMatrix();
		double dist[][] = new double[validationSet.numInstances()][validationSet
				.numClasses()];

		int countFN = 0;
		int countFP = 0;
		int countTN = 0;
		int countTP = 0;
		double pred = 0.0d;
		double actual = 0.0d;
		int i = -1;

		for (Instance instance : validationSet) {
			dist[++i] = m.distributionForInstance(instance);

			pred = dist[i][0] >= dist[i][1] ? 0 : 1;
			// pred = dist[i][1] > 0 ? 1 : 0;
			actual = instance.classValue();
			predInst.add(new InputPredictionInstance((int) actual, (int) pred,
					dist[i]));

			if (pred != actual) {
				if (pred == 1)
					countFP++;
				else
					countFN++;
			} else {
				if (pred == 1)
					countTP++;
				else
					countTN++;
			}
		}

		int mat[][] = { { countTN, countFP }, { countFN, countTP } };
		confMatrix.setMatrix(mat);
		return confMatrix;
		// try {
		// dat.serialize(MetaConstants.NEW_FILE_PATH);
		// } catch (IOException e) {
		// e.printStackTrace();
		// }
	}

	public ModelParams optimize() throws Exception {
		int iterCount = 0;
		divergence = KLDivergenceCalculator.calculate(cState, tState);
		LOGGER.log(Level.FINE, "KL Divergence : " + divergence);
		// ((DivergenceChart) BeanFinder
		// .findBean(MetaConstants.BEAN_DIVERGENCE_CHART)).update(
		// iterCount, divergence);
		double delta[] = null;
		double theta[] = data.getParams().getParams();
		double newDivergence = 0.0d;
		double initialStep = data.getInitialStepSize();
		boolean optimized = false;
		do {
			delta = computeGradient(theta);
			for (double step = initialStep; iterCount < data.getMaxIterations(); step /= 10) {
				LOGGER.log(Level.FINE, "Step size : " + step);
				System.out.println("KL divergence before taking a step : "
						+ KLDivergenceCalculator.calculate(cState, tState));
				for (int i = 0; i < delta.length; i++) {
					if (Double.isNaN(delta[i]))
						System.out.print("delta is NAN!!" + delta[i] + ", ");
					theta[i] -= step * delta[i];
					System.out.print((theta[i]) + ", ");
				}
				System.out.println();
				getNewState(theta);
				iterCount++;
				newDivergence = KLDivergenceCalculator
						.calculate(cState, tState);
				System.out.println("KL Divergence : " + newDivergence);
				// ((DivergenceChart) BeanFinder
				// .findBean(MetaConstants.BEAN_DIVERGENCE_CHART)).update(
				// iterCount, newDivergence);
				if (newDivergence < divergence) {
					divergence = newDivergence;
					initialStep = step;
					break;
				}
				// before taking the next step size, reset theta to what it was
				for (int i = 0; i < delta.length; i++)
					theta[i] += step * delta[i];
				getNewState(theta); // Added for checking. Not required.
			}
			optimized = optimized(theta);
		} while (!optimized && iterCount < data.getMaxIterations());
		ModelParams optimParams = null;
		if (optimized) {
			optimParams = new ModelParams();
			optimParams.setParams(theta);
		}
		return optimParams;
	}

	private void getNewState(double[] theta) throws Exception {
		ModelParams params = new ModelParams();
		params.setParams(theta);
		newData = classifier.computeNewState(params);
		newData.setProxy(classifier);
		cState = CurrentState.createCurrentState(newData.getPredInstances());
		// TargetStateCalculator tstateCalc = new TargetStateCalculator(newData,
		// cState);
		// tState = tstateCalc.calculate();
	}

	protected boolean optimized(double[] theta) throws Exception {
		ConfusionMatrix newConf = newData.getConfMatrix();
		// newConf.display();
		int[][] nc = newConf.getMatrix();
		int[][] c = data.getConfMatrix().getMatrix();
		int[][] b = data.getBiasMatrix().getMatrix();
		boolean blnOptim = true;
		for (int i = 0; i < numLabels; i++)
			for (int j = 0; j < numLabels; j++)
				if (b[i][j] > c[i][j])
					blnOptim = nc[i][j] >= b[i][j];
				else if (b[i][j] < c[i][j])
					blnOptim = nc[i][j] <= b[i][j];

		if (blnOptim) {
			// try {
			// System.out.println("Serializing data after optimization...");
			// newData.serialize(MetaConstants.OPTIMIZED_FILE_PATH);
			// } catch (IOException e) {
			// e.printStackTrace();
			// }
			data.setConfMatrix(newConf);
			data.setPredInstances(newData.getPredInstances());
			data.setParams(newData.getParams());
		}

		return blnOptim;
	}

	private void displayMat(Matrix mat) {
		double[] vals = mat.getRowPackedCopy();
		for (int i = 0; i < vals.length; i++)
			System.out.print(vals[i] + ", ");
	}

	public void setDebug(boolean debug) {
		m_Debug = debug;
	}
}
