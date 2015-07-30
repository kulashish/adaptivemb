package in.ac.iitb.cse.qh.marti;

import in.ac.iitb.cse.qh.classifiers.ModifiedLogistic;
import in.ac.iitb.cse.qh.data.ConfusionMatrix;
import in.ac.iitb.cse.qh.data.CurrentState;
import in.ac.iitb.cse.qh.data.InputData;
import in.ac.iitb.cse.qh.data.ModelParams;
import in.ac.iitb.cse.qh.data.TargetState;
import in.ac.iitb.cse.qh.meta.ClassifierProxy;
import in.ac.iitb.cse.qh.meta.Optimizer;
import in.ac.iitb.cse.qh.meta.TargetStateCalculator;
import in.ac.iitb.cse.qh.util.MetaConstants;
import in.ac.iitb.cse.qh.util.StdRandom;
import in.ac.iitb.cse.qh.util.WekaUtil;

import java.util.logging.Level;
import java.util.logging.Logger;

import weka.core.Instance;
import weka.core.Instances;

public class AdaptiveMartiNode {
	private static final Logger LOGGER = Logger
			.getLogger(AdaptiveMartiNode.class.getName());

	private int number = 0;
	private double beta = 0.0d;
	private ClassifierProxy cProxy;
	private NodeInputData inData;
	private NodeOutputData outData;
	private ModifiedLogistic classifier;
	private boolean blnTrained;
	private AdaptiveMartiLevel level;
	private boolean frozen;
	private int frozenLabel = 0;

	public AdaptiveMartiNode(Instances trainInstances,
			Instances holdoutInstances, AdaptiveMartiLevel level, int number) {
		inData = new NodeInputData(trainInstances, holdoutInstances);
		outData = new NodeOutputData(trainInstances.numInstances(),
				holdoutInstances.numInstances());
		cProxy = new ClassifierProxy();
		this.level = level;
		this.number = number;
	}

	public AdaptiveMartiNode(Instances trainInstances,
			Instances holdoutInstances, Instances testInstances,
			AdaptiveMartiLevel level, int number) {
		this(trainInstances, holdoutInstances, level, number);
		inData.setTestInstances(testInstances);
	}

	public void display() {
		LOGGER.log(Level.INFO,
				"Node number: " + number + " level: " + level.getLevelNumber());
		if (inData.trainingInstances != null)
			LOGGER.log(Level.INFO, "Training instances : "
					+ inData.trainingInstances.size());
		if (inData.holdoutInstances != null)
			LOGGER.log(Level.INFO, "Holdout instances : "
					+ inData.holdoutInstances.size());
		if (inData.testInstances != null)
			LOGGER.log(Level.INFO,
					"Test instances : " + inData.testInstances.size());
		if (outData.trainConfusionMatrix != null) {
			LOGGER.log(Level.INFO, "Training conf matrix: ");
			outData.trainConfusionMatrix.display();
		}
		if (outData.holdoutConfusionMatrix != null) {
			LOGGER.log(Level.INFO, "Holdout conf matrix: ");
			outData.holdoutConfusionMatrix.display();
		}
		LOGGER.log(Level.INFO, "Frozen : " + isFrozen());
		if (isFrozen())
			LOGGER.log(Level.INFO, "Frozen Label: " + frozenLabel);
	}

	public int getNumber() {
		return number;
	}

	public void setNumber(int number) {
		this.number = number;
	}

	public double getBeta() {
		return beta;
	}

	public void setBeta(double beta) {
		this.beta = beta;
	}

	public NodeInputData getInData() {
		return inData;
	}

	public NodeOutputData getOutData() {
		return outData;
	}

	public boolean isFrozen() {
		return frozen;
	}

	public void setFrozen(boolean frozen) {
		this.frozen = frozen;
	}

	public AdaptiveMartiLevel getLevel() {
		return level;
	}

	public void setLevel(AdaptiveMartiLevel level) {
		this.level = level;
	}

	public void train() throws Exception {
		if (MetaConstants.BALANCE_TRAINING_DATA)
			WekaUtil.balanceInstances(inData.trainingInstances);
		InputData inputData = cProxy.computeInitialState(
				inData.trainingInstances, inData.holdoutInstances);

		if (MetaConstants.TUNING) {
			if (level.getLevelNumber() == 0 && number == 0) // root node
				inputData.createDefaultBiasLowFP(0);
			else
				inputData
						.createBias(inData.getTargetFP(), inData.getTargetFN());
			CurrentState currState = CurrentState.createCurrentState(inputData
					.getPredInstances());
			TargetStateCalculator tstateCalc = new TargetStateCalculator(
					inputData, currState);
			TargetState targetState = tstateCalc.calculate();
			Optimizer optimizer = new Optimizer(inputData, currState,
					targetState, cProxy);
			ModelParams params = null;
			try {
				params = optimizer.optimize2();
			} catch (Exception e) {
				e.printStackTrace();
				params = null;
			}
			classifier = cProxy.trainModel(params);
		} else
			classifier = cProxy.getClassifier();

		blnTrained = true;
	}

	public void build() throws Exception {
		checkFreezeNode();
		if (!isFrozen())
			train();
		outData = new NodeOutputData(inData.trainingInstances.numInstances(),
				inData.holdoutInstances.numInstances());
		classify(inData.trainingInstances, outData.trainProb, true);
		// outData.trainConfusionMatrix.display();
		// Freeze the node if the effective train advantage is 0
		if (outData.trainAdvantage == 0) {
			setFrozen(true);
			frozenLabel = outData.trainConfusionMatrix.getTp() != 0 ? 1 : 0;
		}
		classify(inData.holdoutInstances, outData.holdoutProb, false);
	}

	private void checkFreezeNode() {
		int numInstancesPerClass[] = null;
		if (null != inData.trainingInstances)
			numInstancesPerClass = inData.trainingInstances
					.attributeStats(inData.trainingInstances.classIndex()).nominalCounts;
		if (numInstancesPerClass[0] == 0) {
			frozenLabel = 1;
			setFrozen(true);
			LOGGER.log(Level.INFO, "Level: " + getLevel().getLevelNumber()
					+ " Freezing node " + number + " with label " + frozenLabel);
		} else if (numInstancesPerClass[1] == 0) {
			frozenLabel = 0;
			setFrozen(true);
		}
	}

	private void classify(Instances instances, double[] prob, boolean blnTrain)
			throws Exception {
		// LOGGER.log(Level.INFO,
		// "Classifying instances :" + instances.numInstances());
		int[][] conf = isFrozen() ? WekaUtil.classify(instances, frozenLabel)
				: WekaUtil.classify(classifier, instances, prob);
		ConfusionMatrix confusionMatrix = new ConfusionMatrix(conf);
		if (blnTrain) {
			outData.trainConfusionMatrix = confusionMatrix;
			// outData.trainAdvantage = Math.min(confusionMatrix.getAdvPos(),
			// confusionMatrix.getAdvNeg());
			outData.trainAdvantage = confusionMatrix.getMinAdv();
		} else {
			outData.holdoutConfusionMatrix = confusionMatrix;
			// outData.holdoutAdvantage = Math.min(confusionMatrix.getAdvPos(),
			// confusionMatrix.getAdvNeg());
			outData.holdoutAdvantage = confusionMatrix.getMinAdv();
		}
		// LOGGER.log(Level.INFO, "Confusion Matrix:");
		// confusionMatrix.display();
	}

	public void route(double levelAdvantage) {
		route(levelAdvantage, inData.trainingInstances, outData.trainProb,
				(short) 0);
		route(levelAdvantage, inData.holdoutInstances, outData.holdoutProb,
				(short) 1);
	}

	private void route(double levelAdvantage, Instances instances,
			double[] prob, short iType) {
		Instance instance = null;
		int iNextLevelNodeIndex = 0;
		double dNextLevelNodeIndex = 0.0d;
		double p = 0.0d;
		boolean blnFlag = false;
		StdRandom.init();
		double h = 0.0d;
		for (int iInstance = 0; iInstance < instances.numInstances(); iInstance++) {
			instance = instances.get(iInstance);
			h = prob[iInstance] * 2 - 1.0; // scale prob so that it is in [-1,
											// 1]
			dNextLevelNodeIndex = (beta + levelAdvantage * h) * 2
					/ levelAdvantage;
			// System.out.println("Next level node index: " +
			// dNextLevelNodeIndex);
			iNextLevelNodeIndex = (int) Math.floor(dNextLevelNodeIndex);
			p = dNextLevelNodeIndex - iNextLevelNodeIndex;
			blnFlag = StdRandom.bernoulli(p);
			iNextLevelNodeIndex += blnFlag ? 1 : 0;
			if (iType == 2 && !level.nodePresent(iNextLevelNodeIndex))
				iNextLevelNodeIndex += blnFlag ? -1 : 1;
			level.routeInstance(iNextLevelNodeIndex, instance, iType);
		}

	}

	public void addInstance(Instance instance, short iType) {
		switch (iType) {
		case 0:
			inData.trainingInstances.add(instance);
			break;
		case 1:
			inData.holdoutInstances.add(instance);
			break;
		case 2:
			inData.testInstances.add(instance);
			break;
		}
	}

	public void classifyRouteTestInstances(double levelAdvantage)
			throws Exception {
		LOGGER.log(Level.INFO, "Level: " + level.getLevelNumber() + " Node : "
				+ getNumber() + " Frozen: " + isFrozen());
		outData.testProb = new double[inData.testInstances.numInstances()];
		int[][] conf = isFrozen() ? WekaUtil.classify(inData.testInstances,
				frozenLabel) : WekaUtil.classify(classifier,
				inData.testInstances, outData.testProb);
		outData.testConfusionMatrix = new ConfusionMatrix(conf);
		if (!isFrozen())
			route(levelAdvantage, inData.testInstances, outData.testProb,
					(short) 2);
	}

	public void classifyFinalLevelTestInstances() {
		outData.testProb = new double[inData.testInstances.numInstances()];
		outData.testConfusionMatrix = new ConfusionMatrix(WekaUtil.classify(
				inData.testInstances, Math.signum(beta) < 0 ? 0 : 1));
	}

	class NodeInputData {
		Instances trainingInstances;
		Instances holdoutInstances;
		Instances testInstances;
		double targetGammaMinus;
		double targetGammaPlus;

		public NodeInputData(Instances train, Instances holdout) {
			trainingInstances = new Instances(train);
			trainingInstances
					.setClassIndex(trainingInstances.numAttributes() - 1);
			holdoutInstances = new Instances(holdout);
			holdoutInstances
					.setClassIndex(holdoutInstances.numAttributes() - 1);
			// testInstances = new Instances(test);
			// testInstances.setClassIndex(testInstances.numAttributes() - 1);
		}

		public void setTestInstances(Instances testInstances) {
			this.testInstances = new Instances(testInstances);
			testInstances.setClassIndex(testInstances.numAttributes() - 1);
		}

		public int getTargetFN() {
			// TODO Auto-generated method stub
			return 0;
		}

		public int getTargetFP() {
			// TODO Auto-generated method stub
			return 0;
		}
	}

	class NodeOutputData {
		ConfusionMatrix trainConfusionMatrix;
		ConfusionMatrix holdoutConfusionMatrix;
		ConfusionMatrix testConfusionMatrix;
		double[] trainProb;
		double[] holdoutProb;
		double[] testProb;
		double trainAdvantage;
		double holdoutAdvantage;

		public NodeOutputData(int numTrainInstances, int numHoldoutInstances) {
			trainProb = new double[numTrainInstances];
			holdoutProb = new double[numHoldoutInstances];
		}
	}
}
