package in.ac.iitb.cse.qh.marti;

import in.ac.iitb.cse.qh.data.ConfusionMatrix;
import in.ac.iitb.cse.qh.data.CurrentState;
import in.ac.iitb.cse.qh.data.InputData;
import in.ac.iitb.cse.qh.data.ModelParams;
import in.ac.iitb.cse.qh.data.TargetState;
import in.ac.iitb.cse.qh.meta.ClassifierProxy;
import in.ac.iitb.cse.qh.meta.Optimizer;
import in.ac.iitb.cse.qh.meta.TargetStateCalculator;
import in.ac.iitb.cse.qh.util.MessageConstants;
import in.ac.iitb.cse.qh.util.MetaConstants;

import java.util.logging.Level;
import java.util.logging.Logger;

import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class MartiNode {

	protected int level, number;
	private NodeInputData inData;
	private NodeOutputData outData;

	public MartiNode leftNode;
	public MartiNode rightNode;
	public MartiNode leftParent;
	public MartiNode rightParent;
	protected Logistic m_log;
	private ClassifierProxy classifierProxy;

	protected boolean isTested, isTrained, isTrainTested, isFinal, isFreezed,
			freezeLabel;
	protected double prob_pos, prob_neg;

	public NodeInputData getInData() {
		return inData;
	}

	public NodeOutputData getOutData() {
		return outData;
	}

	double getProbPos() {
		return prob_pos;
	}

	double getProbNeg() {
		return prob_neg;
	}

	public int getLevel() {
		return level;
	}

	public int getNumber() {
		return number;
	}

	public boolean classifySingleInstance(Instance inst) {

		boolean bLabel = false;
		try {
			double classLabel = m_log.classifyInstance(inst);
			if (Utils.eq(classLabel, 1.0))
				bLabel = true;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return bLabel;
	}

	public void computeProbabilities() {

		if (null != rightParent
				&& null != rightParent.outData.holdoutConfusionMatrix) {
			prob_pos += rightParent.prob_pos
					* (0.5 - rightParent.outData.holdoutConfusionMatrix
							.getAdvPos());
			prob_neg += rightParent.prob_neg
					* (0.5 + rightParent.outData.holdoutConfusionMatrix
							.getAdvNeg());
		}
		if (null != leftParent
				&& null != leftParent.outData.holdoutConfusionMatrix) {
			prob_pos += leftParent.prob_pos
					* (0.5 + leftParent.outData.holdoutConfusionMatrix
							.getAdvPos());
			prob_neg += leftParent.prob_neg
					* (0.5 - leftParent.outData.holdoutConfusionMatrix
							.getAdvNeg());
		}
	}

	public ConfusionMatrix freezeEmptyNode() {
		ConfusionMatrix mat = null;
		LOGGER.log(Level.WARNING, "Node v" + getNodeLevel() + getNodeNumber()
				+ " received no training instances");
		LOGGER.log(Level.WARNING, "Node " + getNodeLevel() + " "
				+ getNodeNumber() + " not built");
		computeProbabilities();
		if (getProbPos() >= getProbNeg()) {
			freezeEmptyNode(true);
			mat = new ConfusionMatrix(new int[][] {
					{ 0, getInData().numNegativeHoldoutInstances() },
					{ 0, getInData().numPositiveHoldoutInstances() } });
		} else {
			freezeEmptyNode(false);
			mat = new ConfusionMatrix(new int[][] {
					{ getInData().numNegativeHoldoutInstances(), 0 },
					{ getInData().numPositiveHoldoutInstances(), 0 } });
		}
		return mat;
	}

	// returns true and freezes the node else just returns false

	void freezeEmptyNode(boolean label) {
		isFreezed = true;
		freezeLabel = label;
	}

	boolean freezeNode(double epsilon, int numLevels) {

		double t = epsilon / (numLevels * (numLevels + 1));

		// if(MetaConstants.FREEZE_HARDCODE)
		// t = MetaConstants.FREEZE_VALUE;

		if (prob_pos < t) {
			isFreezed = true;
			freezeLabel = false;
			this.outData.holdoutConfusionMatrix = new ConfusionMatrix(
					new int[][] { { inData.numNegativeHoldoutInstances(), 0 },
							{ inData.numPositiveHoldoutInstances(), 0 } });
		} else if (prob_neg < t) {
			isFreezed = true;
			freezeLabel = true;
			this.outData.holdoutConfusionMatrix = new ConfusionMatrix(
					new int[][] { { 0, inData.numNegativeHoldoutInstances() },
							{ 0, inData.numPositiveHoldoutInstances() } });
		}
		return isFreezed;
	}

	public boolean isNodeFreezed() {
		return this.isFreezed;
	}

	public boolean getFreezedNodeLabel() {
		return this.freezeLabel;
	}

	private static final Logger LOGGER = Logger.getLogger(MartiNode.class
			.getName());
	static {
		LOGGER.setLevel(Level.FINE);
	}

	public boolean isTrained() {
		return isTrained;
	}

	public void display() {
		LOGGER.log(Level.INFO, "Displaying Node " + level + ", " + number);
		inData.display();
		outData.display();
		LOGGER.log(Level.INFO, "Prob Pos " + prob_pos + ", Prob Neg "
				+ prob_neg);
	}

	public ConfusionMatrix getHoldoutConfusionMatrix() {
		return outData.holdoutConfusionMatrix;
	}

	public int numTrInstances() {
		if (inData.m_train_Instances == null)
			return 0;
		else
			return inData.m_train_Instances.numInstances();
	}

	public void setFinal() {
		isFinal = true;
	}

	public boolean getIsFinal() {
		return isFinal;
	}

	// use for creating root node
	public MartiNode(Instances train, Instances test) {
		isTrained = false;
		isTested = false;
		isTrainTested = false;
		isFinal = false;
		isFreezed = false;
		number = 0;
		level = 0;
		leftParent = null;
		rightParent = null;
		leftNode = null;
		rightNode = null;
		inData = new NodeInputData(train, test);
		outData = new NodeOutputData();
		prob_pos = 1;
		prob_neg = 1;
		// m_log = new Logistic();
		// m_log = new ModifiedLogistic();
		classifierProxy = new ClassifierProxy();
	}

	// use for creating other nodes
	public MartiNode() {
		isTrained = false;
		isTested = false;
		isTrainTested = false;
		isFinal = false;
		leftParent = null;
		rightParent = null;
		leftNode = null;
		rightNode = null;
		inData = new NodeInputData();
		outData = new NodeOutputData();
		// m_log = new Logistic();
		// m_log = new ModifiedLogistic();
		classifierProxy = new ClassifierProxy();
		prob_pos = 0;
		prob_neg = 0;
	}

	public int getNodeLevel() {
		return level;
	}

	public int getNodeNumber() {
		return number;
	}

	private void setNodeLevel(int n) {
		level = n;
	}

	private void setNodeNumber(int i) {
		number = i;
	}

	public void setInstancesFromParent() {
		if (leftParent != null && !leftParent.isNodeFreezed()) {
			LOGGER.log(Level.INFO, "Setting instances from left parent ...");
			setInstancesFromLeftParent(leftParent.getPositiveTrainInstances(),
					leftParent.getPositiveTestInstances());
		}
		if (rightParent != null && !rightParent.isNodeFreezed()) {
			LOGGER.log(Level.INFO, "Setting instances from right parent ...");
			setInstancesFromRightParent(
					rightParent.getNegativeTrainInstances(),
					rightParent.getNegativeTestInstances());
		}

		mergeTrainInstances();
		mergeTestInstances();
	}

	public void setInstancesFromLeftParent(Instances train, Instances test) {
		if (isFinal) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_FINAL);
			return;

		}
		inData.m_train_Instances_left = new Instances(train);
		inData.m_train_Instances_left
				.setClassIndex(inData.m_train_Instances_left.numAttributes() - 1);
		inData.m_test_Instances_left = new Instances(test);
		inData.m_test_Instances_left.setClassIndex(inData.m_test_Instances_left
				.numAttributes() - 1);
	}

	public void setInstancesFromRightParent(Instances train, Instances test) {
		if (isFinal) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_FINAL);
			return;

		}
		inData.m_train_Instances_right = new Instances(train);
		inData.m_train_Instances_right
				.setClassIndex(inData.m_train_Instances_right.numAttributes() - 1);
		inData.m_test_Instances_right = new Instances(test);
		inData.m_test_Instances_right
				.setClassIndex(inData.m_test_Instances_right.numAttributes() - 1);
	}

	public void mergeTestInstances() {

		if (isFinal) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_FINAL);
			return;

		}
		if (inData.m_test_Instances_left == null
				&& inData.m_test_Instances_right == null)
			return;
		if (inData.m_test_Instances_left == null) {
			inData.m_test_Instances = new Instances(
					inData.m_test_Instances_right);
			inData.m_test_Instances.setClassIndex(inData.m_test_Instances
					.numAttributes() - 1);
			return;
		}
		inData.m_test_Instances = new Instances(inData.m_test_Instances_left);
		if (inData.m_train_Instances_right == null) {
			inData.m_test_Instances.setClassIndex(inData.m_test_Instances
					.numAttributes() - 1);
			return;
		}
		for (int i = 0; i < inData.m_test_Instances_right.numInstances(); i++) {
			Instance instance = inData.m_test_Instances_right.instance(i);
			inData.m_test_Instances.add(instance);
		}
		inData.m_test_Instances.setClassIndex(inData.m_test_Instances
				.numAttributes() - 1);
	}

	public void mergeTrainInstances() {

		if (isFinal) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_FINAL);
			return;

		}
		if (inData.m_train_Instances_left == null
				&& inData.m_train_Instances_right == null)
			return;
		if (inData.m_train_Instances_left == null) {
			inData.m_train_Instances = new Instances(
					inData.m_train_Instances_right);
			inData.m_train_Instances.setClassIndex(inData.m_train_Instances
					.numAttributes() - 1);
			return;
		}
		inData.m_train_Instances = new Instances(inData.m_train_Instances_left);
		if (inData.m_train_Instances_right == null) {
			inData.m_train_Instances.setClassIndex(inData.m_train_Instances
					.numAttributes() - 1);
			return;
		}
		for (int i = 0; i < inData.m_train_Instances_right.numInstances(); i++) {
			Instance instance = inData.m_train_Instances_right.instance(i);
			inData.m_train_Instances.add(instance);
		}
		inData.m_train_Instances.setClassIndex(inData.m_train_Instances
				.numAttributes() - 1);
	}

	public void train() {
		if (isFinal) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_FINAL);
			return;
		}
		try {
			LOGGER.log(Level.INFO, "Training classifier on "
					+ inData.m_train_Instances.numInstances() + " Instances");
			if (MetaConstants.BALANCE_TRAINING_DATA)
				balanceTrain();
			// 1. Compute initial state
			InputData inputData = classifierProxy.computeInitialState(
					inData.m_train_Instances, inData.m_test_Instances);
			if (MetaConstants.TUNING) {
//				if (level == 0 && number == 0)
//					inputData.createDefaultBiasLowFP(0);
					inputData.createBias();
//				else
//					inputData.createBias(inData.getTargetFP(),
//							inData.getTargetFN());
				// 2. Create current state
				CurrentState currState = CurrentState
						.createCurrentState(inputData.getPredInstances());
				// 3. Create target state
				TargetStateCalculator tstateCalc = new TargetStateCalculator(
						inputData, currState);
				TargetState targetState = tstateCalc.calculate();
				// 4. Run optimizer
				// Optimizer optimizer = new DefensiveOptimizer(inputData,
				// currState,
				// targetState, classifierProxy);
				Optimizer optimizer = new Optimizer(inputData, currState,
						targetState, classifierProxy);
				ModelParams params = null;
				try {
					params = optimizer.optimize2();
				} catch (Exception e) {
					e.printStackTrace();
					params = null;
				}
				// 5. Train the ModifiedLogistic model using tuned
				// hyperparameters
				m_log = classifierProxy.trainModel(params);
			} else
				m_log = classifierProxy.getClassifier();

			isTrained = true;
		} catch (Exception e) {
			LOGGER.log(Level.SEVERE, "Exception occured");
			e.printStackTrace();
		}
	}

	public void classifyTest() {
		if (isFinal) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_FINAL);
			return;
		}
		if (!isTrained) {
			LOGGER.log(Level.WARNING, "Classifier not trained");
			return;
		} else {
			LOGGER.log(Level.INFO, "Evaluating holdout data on "
					+ inData.m_test_Instances.numInstances() + " Instances");
			outData.m_positive_Test_Instances = new Instances(
					inData.m_test_Instances, 0);
			outData.m_negative_Test_Instances = new Instances(
					inData.m_test_Instances, 0);
			outData.holdoutConfusionMatrix = classify(inData.m_test_Instances,
					outData.m_positive_Test_Instances,
					outData.m_negative_Test_Instances);
			isTested = true;
		}
	}

	public void classifyTrain() {
		if (isFinal) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_FINAL);
			return;

		}
		if (!isTrained) {
			LOGGER.log(Level.WARNING, "Classifier not trained");
			return;
		} else {
			LOGGER.log(Level.INFO, "Evaluating training data on "
					+ inData.m_train_Instances.numInstances() + " Instances");
			outData.m_positive_Train_Instances = new Instances(
					inData.m_train_Instances, 0);
			outData.m_negative_Train_Instances = new Instances(
					inData.m_train_Instances, 0);
			outData.trainConfusionMatrix = classify(inData.m_train_Instances,
					outData.m_positive_Train_Instances,
					outData.m_negative_Train_Instances);

			isTrainTested = true;
		}
	}

	public Logistic getM_log() {
		return m_log;
	}

	public void setM_log(Logistic m_log) {
		this.m_log = m_log;
	}

	public ConfusionMatrix classify(Instances instancesToClassify,
			Instances positiveInstances, Instances negativeInstances) {
		int tp = 0;
		int tn = 0;
		int fp = 0;
		int fn = 0;

		double classLabel = 0.0d;
		double[] dist = null;
		Instance instance = null;
		for (int i = 0; i < instancesToClassify.numInstances(); i++) {
			instance = instancesToClassify.instance(i);
			try {
				//classLabel = m_log.classifyInstance(instance);
				dist = m_log.distributionForInstance(instance);
				classLabel = dist[0] >= dist[1] ? 0 : 1;
				if (Utils.eq(classLabel, 1.0)) {
					if (MetaConstants.CONFIDENCE_BASED_REWEIGHTING)
						instance.setWeight(dist[0]); // re-weight the instance
														// based
														// on classifier's error
														// in
														// classifying it
					positiveInstances.add(instance);
					if (Utils.eq(instance.classValue(), 1.0))
						tp++;
					else
						fp++;
				} else {
					if (MetaConstants.CONFIDENCE_BASED_REWEIGHTING)
						instance.setWeight(dist[1]); // re-weight the instance
					// based
					// on classifier's error in
					// classifying it
					negativeInstances.add(instance);
					if (Utils.eq(instance.classValue(), 0.0))
						tn++;
					else
						fn++;
				}
			} catch (Exception e) {
				e.printStackTrace();
			}

		}
		return new ConfusionMatrix(new int[][] { { tn, fp }, { fn, tp } });
	}

	// positively classified Test Instances
	public Instances getPositiveTestInstances() {
		if (isFinal) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_FINAL);
			return null;

		}
		if (isTested)
			return outData.m_positive_Test_Instances;
		else
			return null;
	}

	// Positively classified Training Instances
	public Instances getPositiveTrainInstances() {
		if (isFinal) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_FINAL);
			return null;

		}
		if (isTrainTested) {
			return outData.m_positive_Train_Instances;
		} else {
			LOGGER.log(Level.WARNING, "Node is not train tested!");
			return null;
		}
	}

	// Negatively classified Test Instances
	public Instances getNegativeTestInstances() {
		if (isFinal) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_FINAL);
			return null;

		}
		if (isTested)
			return outData.m_negative_Test_Instances;
		else
			return null;
	}

	// Negatively classified Train Instances
	public Instances getNegativeTrainInstances() {
		if (isFinal) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_FINAL);
			return null;

		}
		if (isTrainTested)
			return outData.m_negative_Train_Instances;
		else
			return null;
	}

	public void balanceTrain() {
		if (isFinal) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_FINAL);
			return;

		}
		int p = 0, n = 0;
		for (int i = 0; i < inData.m_train_Instances.numInstances(); i++) {
			Instance instance = inData.m_train_Instances.instance(i);
			if (Utils.eq(instance.classValue(), 1.0))
				p++;
			else
				n++;
		}
		double r = (1.0d * p) / n;
		for (int i = 0; i < inData.m_train_Instances.numInstances(); i++) {
			if (Utils
					.eq(inData.m_train_Instances.instance(i).classValue(), 0.0))
				inData.m_train_Instances.instance(i).setWeight(r);
		}
	}

	public MartiNode addLeftNode() {
		if (isFinal) {
			LOGGER.log(Level.WARNING, "Node on the final level");
			return null;

		}
		leftNode = new MartiNode();
		leftNode.setNodeLevel(level + 1);
		leftNode.setNodeNumber(number);
		leftNode.rightParent = this;
		return leftNode;
	}

	public MartiNode addRightNode() {
		if (isFinal) {
			LOGGER.log(Level.WARNING, "Node on the final level");
			return null;

		}
		rightNode = new MartiNode();
		rightNode.setNodeLevel(level + 1);
		rightNode.setNodeNumber(number + 1);
		rightNode.leftParent = this;
		return rightNode;
	}

	public MartiNode getNextNode() {
		return level == number ? null : rightParent.rightNode;
	}

	public boolean isTrain() {

		if (inData.m_train_Instances == null)
			return false;
		int numInstancesPerClass[] = inData.m_train_Instances
				.attributeStats(inData.m_train_Instances.classIndex()).nominalCounts;
		return numInstancesPerClass[0] > 0 && numInstancesPerClass[1] > 0;
	}

	class NodeInputData {
		public NodeInputData() {

		}

		public int getTargetFP() {
			targetFp = (int) (numNegativeHoldoutInstances() * (0.5 - targetGammaMinus));
			return targetFp;
		}

		public int getTargetFN() {
			targetFn = (int) (numPositiveHoldoutInstances() * (0.5 - targetGammaPlus));
			return targetFn;
		}

		public NodeInputData(Instances train, Instances test) {
			m_train_Instances = new Instances(train);
			m_train_Instances
					.setClassIndex(m_train_Instances.numAttributes() - 1);
			m_test_Instances = new Instances(test);
			m_test_Instances
					.setClassIndex(m_test_Instances.numAttributes() - 1);
		}

		public int numPositiveTrainInstances() {
			int numInstancesPerClass[] = null;
			if (null != m_train_Instances)
				numInstancesPerClass = m_train_Instances
						.attributeStats(m_train_Instances.classIndex()).nominalCounts;

			return null != numInstancesPerClass ? numInstancesPerClass[1] : 0;
		}

		public int numPositiveHoldoutInstances() {
			int numInstancesPerClass[] = null;
			if (null != m_test_Instances)
				numInstancesPerClass = m_test_Instances
						.attributeStats(m_test_Instances.classIndex()).nominalCounts;

			return null != numInstancesPerClass ? numInstancesPerClass[1] : 0;
		}

		public int numNegativeHoldoutInstances() {
			int numInstancesPerClass[] = null;
			if (null != m_test_Instances)
				numInstancesPerClass = m_test_Instances
						.attributeStats(m_test_Instances.classIndex()).nominalCounts;

			return null != numInstancesPerClass ? numInstancesPerClass[0] : 0;
		}

		public int numNegativeTrainInstances() {
			int numInstancesPerClass[] = null;
			if (null != m_train_Instances)
				numInstancesPerClass = m_train_Instances
						.attributeStats(m_train_Instances.classIndex()).nominalCounts;

			return null != numInstancesPerClass ? numInstancesPerClass[0] : 0;
		}

		public void display() {
			LOGGER.log(Level.INFO, "Number of +ve training instances :"
					+ numPositiveTrainInstances());
			LOGGER.log(Level.INFO, "Number of -ve training instances :"
					+ numNegativeTrainInstances());

			LOGGER.log(
					Level.INFO,
					"Number of holdout instances : "
							+ (null != m_test_Instances ? m_test_Instances
									.numInstances() : 0));

			LOGGER.log(Level.INFO, "Target gamma- = " + targetGammaMinus);
			LOGGER.log(Level.INFO, "Target gamma+ = " + targetGammaPlus);
			LOGGER.log(Level.INFO, "Target FP = " + targetFp);
			LOGGER.log(Level.INFO, "Target FN = " + targetFn);
		}

		Instances m_train_Instances_left;// Training Instances received from
		// left parent
		Instances m_train_Instances_right;// Training Instances received
		// from right parent
		Instances m_test_Instances_left;// Test Instances received from left
		// parent
		Instances m_test_Instances_right;// Test Instances received from
		// right parent
		Instances m_train_Instances;// Merged m_train_Instances_left +
		// m_train_Instances_right
		Instances m_test_Instances;// Merged m_test_Instances_left +
		// m_test_Instances_right

		double targetGammaMinus;
		double targetGammaPlus;
		int targetFp;
		int targetFn;
	}

	class NodeOutputData {
		ConfusionMatrix trainConfusionMatrix;
		ConfusionMatrix holdoutConfusionMatrix;
		Instances m_positive_Test_Instances;// positive test instances after
		// classification
		Instances m_negative_Test_Instances;// negative test instances after
		// classification
		Instances m_positive_Train_Instances;// positive training instances
		// after classification
		Instances m_negative_Train_Instances;// negative training instances
												// after classification
		double errorMinus;
		double errorPlus;

		public NodeOutputData() {
			trainConfusionMatrix = new ConfusionMatrix();
			holdoutConfusionMatrix = new ConfusionMatrix();
		}

		public void display() {
			LOGGER.log(
					Level.INFO,
					"Number of training instances classified positive: "
							+ (null != m_positive_Train_Instances ? m_positive_Train_Instances
									.numInstances() : 0));
			LOGGER.log(
					Level.INFO,
					"Number of training instances classified negative: "
							+ (null != m_negative_Train_Instances ? m_negative_Train_Instances
									.numInstances() : 0));
			LOGGER.log(Level.INFO, "Training confusion matrix:");
			if (null != trainConfusionMatrix)
				trainConfusionMatrix.display();

			LOGGER.log(
					Level.INFO,
					"Number of holdout instances classified positive: "
							+ (null != m_positive_Test_Instances ? m_positive_Test_Instances
									.numInstances() : 0));
			LOGGER.log(
					Level.INFO,
					"Number of holdout instances classified negative: "
							+ (null != m_negative_Test_Instances ? m_negative_Test_Instances
									.numInstances() : 0));

			if (null != holdoutConfusionMatrix) {
				LOGGER.log(Level.INFO, "Holdout confusion matrix:");
				holdoutConfusionMatrix.display();

				LOGGER.log(Level.INFO, "Gamma minus : "
						+ holdoutConfusionMatrix.getAdvNeg());
				LOGGER.log(Level.INFO,
						"Gamma plus : " + holdoutConfusionMatrix.getAdvPos());
			}
		}

		public int numPositiveTrainInstances() {
			return m_positive_Train_Instances.size();
		}

		public int numNegativeTrainInstances() {
			return m_negative_Train_Instances.size();
		}

	}

	public void setTargetGamma() {

		if (leftParent == null && rightParent == null) {
			inData.targetGammaMinus = outData.holdoutConfusionMatrix
					.getAdvNeg();
			inData.targetGammaPlus = outData.holdoutConfusionMatrix.getAdvPos();
		} else {
			if (leftParent != null) {
				inData.targetGammaMinus = leftParent.inData.targetGammaMinus;
				inData.targetGammaPlus = leftParent.inData.targetGammaPlus;
			} else {
				inData.targetGammaMinus = rightParent.inData.targetGammaMinus;
				inData.targetGammaPlus = rightParent.inData.targetGammaPlus;

			}
		}

	}

	// public void setTargetGamma() {
	//
	// if (null != leftParent && null != rightParent) {
	// inData.targetGammaMinus = Math.max(
	// leftParent.outData.holdoutConfusionMatrix.getAdvNeg(),
	// rightParent.outData.holdoutConfusionMatrix.getAdvNeg());
	// inData.targetGammaPlus = Math.max(
	// leftParent.outData.holdoutConfusionMatrix.getAdvPos(),
	// rightParent.outData.holdoutConfusionMatrix.getAdvPos());
	// } else {
	// inData.targetGammaMinus = null != leftParent
	// && null != leftParent.outData.holdoutConfusionMatrix ?
	// leftParent.outData.holdoutConfusionMatrix
	// .getAdvNeg()
	// : (null != rightParent.outData.holdoutConfusionMatrix ?
	// rightParent.outData.holdoutConfusionMatrix
	// .getAdvNeg() : 0);
	// inData.targetGammaPlus = null != leftParent
	// && null != leftParent.outData.holdoutConfusionMatrix ?
	// leftParent.outData.holdoutConfusionMatrix
	// .getAdvPos()
	// : (null != rightParent.outData.holdoutConfusionMatrix ?
	// rightParent.outData.holdoutConfusionMatrix
	// .getAdvPos() : 0);
	// }
	//
	// }
}
