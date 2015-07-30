package in.ac.iitb.cse.qh.meta;

import in.ac.iitb.cse.qh.classifiers.ModifiedLogistic;
import in.ac.iitb.cse.qh.data.ConfusionMatrix;
import in.ac.iitb.cse.qh.data.InputData;
import in.ac.iitb.cse.qh.data.InputPredictionInstance;
import in.ac.iitb.cse.qh.data.ModelParams;
import in.ac.iitb.cse.qh.util.WekaUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.core.Instance;
import weka.core.Instances;

public class ClassifierProxy {
	private static final String CLASS_NAME = "ClassifierProxy";
	private static final Logger logger = Logger.getLogger(CLASS_NAME);
	private ModifiedLogistic mlrClassifier;

	private String tPath;
	private String hPath;
	private Instances trainInstances;
	private Instances holdoutInstances;

	static {
		logger.setLevel(Level.INFO);
	}

	public Instances getholdoutInstances() {
		return holdoutInstances;
	}

	public Instances gettrainInstances() {
		return trainInstances;
	}

	public ModifiedLogistic getClassifier() {
		// if (null == nbClassfier)
		// nbClassfier = new NaiveBayesParam(MetaConstants.TRAIN_FILE_PATH,
		// MetaConstants.TRAIN_FILE_PATH);
		if (null == mlrClassifier) {
			logger.log(Level.FINE, "creating new instance of classifer");
			mlrClassifier = new ModifiedLogistic();
		}
		return mlrClassifier;
	}

	// public double[][] computeJacobian(int instanceIndex, double[] params) {
	// return getClassifier().calcJacobian(instanceIndex);
	// }
	//
	// public ConfusionMatrix computeConfusionMatrix(ModelParams params) {
	// return getClassifier().calcNewState(params.getParams()).getConfMatrix();
	// }
	//

	public ModifiedLogistic trainModel(ModelParams params) throws Exception {
		ModifiedLogistic classifier = getClassifier();
		if (null != params)
			classifier.setHyperparameters(params.getParams());
		classifier.setMaxIts(-1);
		classifier.buildClassifier(trainInstances);
		return classifier;
	}

	public InputData computeInitialState(Instances trainInstances,
			Instances holdoutInstances) throws Exception {
		this.trainInstances = trainInstances;
		this.holdoutInstances = holdoutInstances;
		mlrClassifier = null;
		InputData in = computeNewState(null);
		// validateTraining(in);
		// try {
		// in.serialize(MetaConstants.IN_FILE_PATH);
		// } catch (IOException e) {
		// e.printStackTrace();
		// }
		return in;
	}

	public InputData computeInitialState(String train, String hold)
			throws Exception {
		tPath = train;
		hPath = hold;
		// mlrClassifier = null;
		// InputData in = computeNewState(null);
		// validateTraining(in);
		// try {
		// in.serialize(MetaConstants.IN_FILE_PATH);
		// } catch (IOException e) {
		// e.printStackTrace();
		// }
		// return in;
		return computeInitialState(WekaUtil.getInstances(tPath),
				WekaUtil.getInstances(hPath));
	}

	public InputData computeNewState(ModelParams params) throws Exception {
		String methodName = "computeNewState";
		if (logger.isLoggable(Level.FINER))
			logger.entering(CLASS_NAME, methodName);

		InputData dat = new InputData();
		getClassifier().setMaxIts(-1);
		if (null != params) {
			// for (int i = 0; i < params.getParams().length; i++)
			// System.out.print(params.getParams()[i] + ", ");
			// System.out.println();
			// System.out.println("set hyper parameters...");
			getClassifier().setHyperparameters(params.getParams());
			getClassifier().setMaxIts(-1);
		}
		// else {
		// trainInstances = WekaUtil.getInstances(tPath);
		// holdoutInstances = WekaUtil.getInstances(hPath);
		// }

		ModifiedLogistic ml = getClassifier();
		logger.log(Level.FINE, "building classifier...");
		ml.buildClassifier(trainInstances);
		ModelParams modPar = new ModelParams();
		modPar.setParams(ml.getHyperparameters());
		modPar.setWParams(ml.getWparameters());
		dat.setParams(modPar);

		logger.log(Level.FINE, "validating model...");
		validateModel(dat, holdoutInstances);
		validateTraining(dat);

		if (logger.isLoggable(Level.FINER))
			logger.exiting(CLASS_NAME, methodName);
		return dat;
	}

	/**
	 * Runs the currently trained model on the training data and updates the
	 * training predictions and confusion matrix
	 * 
	 * @param dat
	 * @throws Exception
	 */
	public void validateTraining(InputData dat) throws Exception {
		InputData tempDat = new InputData();
		validateModel(tempDat, trainInstances);
		dat.setTrainPredInstances(tempDat.getPredInstances());
		dat.setTrainConfMatrix(tempDat.getConfMatrix());
		logger.log(Level.FINE, "validate training done");
	}

	private void validateModel(InputData dat, Instances validationSet)
			throws Exception {
		logger.log(Level.FINE, "Number of instances : " + validationSet.size());
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
			dist[++i] = getClassifier().distributionForInstance(instance);

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
		dat.setConfMatrix(confMatrix);
		dat.setPredInstances(predInst);
		// try {
		// dat.serialize(MetaConstants.NEW_FILE_PATH);
		// } catch (IOException e) {
		// e.printStackTrace();
		// }
	}

	/*
	 * Returns gradient of P_i with respect to w computed at w* where P_i is ith
	 * label
	 */
	public double[][] computeJacobian(int i) {
		double[][] jac = getClassifier().computeJacobian(i);
		// for (int a = 0; a < jac.length; a++) {
		// for (int j = 0; j < jac[a].length; j++)
		// System.out.print(jac[a][j] + " ");
		// System.out.println();
		// }
		return jac;
	}

	/*
	 * Covariance matrix C as defined in A. Ng's paper
	 */
	public double[][] getCovariance() {
		double diag[] = getClassifier().getCovariance();
		return getCovariance(diag);
	}

	public double[][] getCovariance(double diag[]) {
		double c[][] = new double[diag.length][diag.length];
		for (int i = 0; i < diag.length; i++)
			for (int j = 0; j < diag.length; j++)
				if (i == j) {
					c[i][j] = Math.exp(diag[i]);
					// if (Double.isNaN(c[i][j]) || Double.isInfinite(c[i][j]))
					logger.log(Level.FINE, c[i][j] + ", ");
				} else
					c[i][j] = 1d;
		// System.out.println();
		return c;
	}

	/*
	 * Hessian of the training log loss evaluated at w*. Refer A. Ng's paper
	 * equation (6)
	 */
	public double[][] getHessian() {
		// System.out.println(this);
		return getClassifier().getHessian();
	}

	/*
	 * Indicator matrix I as defined in A. Ng's paper. Refer equation (6)
	 */
	public double[][] getIndicator() {
		return getClassifier().getIndicator();
	}

	public double[] getWeights() {
		return getClassifier().getWeights();
	}

	public double[] getData(int index) {
		return getClassifier().getData(index);
	}

	public void setTrainingInstances(Instances instances) {
		this.trainInstances = instances;

	}

	public void setHoldoutInstances(Instances holdoutInstances) {
		this.holdoutInstances = holdoutInstances;

	}
}
