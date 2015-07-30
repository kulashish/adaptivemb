package in.ac.iitb.cse.qh.marti;

import in.ac.iitb.cse.qh.data.ConfusionMatrix;
import in.ac.iitb.cse.qh.meta.Optimizer;
import in.ac.iitb.cse.qh.util.MessageConstants;
import in.ac.iitb.cse.qh.util.MetaConstants;
import in.ac.iitb.cse.qh.util.WekaUtil;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

import weka.core.Instances;
import weka.core.Utils;

public class MartiBoost {
	private static final Logger LOGGER = Logger.getLogger(MartiBoost.class
			.getName());
	// private static final String TRAIN_FILE =
	// "/Users/ashish/quickheal/DataSets/UCI/uci-train.arff";
	// private static final String HOLDOUT_FILE =
	// "/Users/ashish/quickheal/DataSets/UCI/uci-holdout.arff";

	private static String TRAIN_FILE = "C:/Users/akulkarni/Documents/Personal/PhD/AAAI/Dataset/spam-train/uci-train-41.arff";
	private static String HOLDOUT_FILE = "C:/Users/akulkarni/Documents/Personal/PhD/AAAI/Dataset/spam-holdout/uci-holdout-41.arff";
	private static String TEST_FILE = "C:/Users/akulkarni/Documents/Personal/PhD/AAAI/Dataset/spam-test/uci-test-1.arff";
	private MartiNode root;
	private int numLevels;
	private ConfusionMatrix gb_matrix;

	static {
		LOGGER.setLevel(Level.INFO);
	}

	public MartiBoost() {

	}

	public MartiBoost(String trainData, String holdoutData, int levels)
			throws Exception {
		this.numLevels = levels;
		Instances trainInstances = WekaUtil.getInstances(trainData);
		Instances holdoutInstances = WekaUtil.getInstances(holdoutData);
		root = new MartiNode(trainInstances, holdoutInstances);
		buildLevel(root);
		root.display();
		gb_matrix = new ConfusionMatrix();
		LOGGER.log(Level.INFO, "Initial global conf matrix :");
		gb_matrix.display();
	}

	public MartiBoost(MartiNode root, int levels, ConfusionMatrix mat) {
		this.root = root;
		this.numLevels = levels;
		this.gb_matrix = mat;
	}

	public MartiNode getRoot() {
		return root;
	}

	public int getNumLevels() {
		return numLevels;
	}

	public void freezeNodesAtALevel(MartiNode node, double epsilon) {
		int n = node.getNodeNumber();
		int l = node.getNodeLevel();
		if (n != 0) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_NOTFIRST);
			return;
		}
		for (int i = 0; i <= l; i++) {
			if (!node.isNodeFreezed() && node.freezeNode(epsilon, numLevels)) {
				LOGGER.log(
						Level.INFO,
						"Node " + node.getNodeLevel() + ","
								+ node.getNodeNumber() + " freezed with label "
								+ node.getFreezedNodeLabel());
				boolean bl = node.getFreezedNodeLabel();
				if (bl) {
					gb_matrix.setFp(gb_matrix.getFp()
							+ node.getInData().numNegativeHoldoutInstances());
					gb_matrix.setTp(gb_matrix.getTp()
							+ node.getInData().numPositiveHoldoutInstances());
				} else {
					gb_matrix.setTn(gb_matrix.getTn()
							+ node.getInData().numNegativeHoldoutInstances());
					gb_matrix.setFn(gb_matrix.getFn()
							+ node.getInData().numPositiveHoldoutInstances());
				}
			}
			node = node.getNextNode();
		}
	}

	public void evaluateTestData(MartiNode root, String testData)
			throws Exception {

		MartiNode curr;
		int tp = 0, fp = 0, tn = 0, fn = 0;
		Instances testInstances = WekaUtil.getInstances(testData);
		for (int i = 0; i < testInstances.numInstances(); i++) {
			curr = root;
			// LOGGER.log(Level.INFO,"Instance class is: ",
			// testInstances.instance(i).classValue());
			// System.out.println("Instance class is: " +
			// testInstances.instance(i).classValue());
			while (true) {
				if (curr.leftNode == null && curr.rightNode == null) {
					int t = curr.getNodeLevel();
					int n = curr.getNodeNumber();
					if (n <= t / 2) {
						if (Utils.eq(testInstances.instance(i).classValue(),
								1.0))
							fn++;
						else
							tn++;

					} else {
						if (Utils.eq(testInstances.instance(i).classValue(),
								1.0))
							tp++;
						else
							fp++;

					}
					break;
				} else if (curr.getIsFinal()) {
					int t = curr.getNodeLevel();
					int n = curr.getNodeNumber();
					if (n <= t / 2) {
						if (Utils.eq(testInstances.instance(i).classValue(),
								1.0))
							fn++;
						else
							tn++;
					} else {
						if (Utils.eq(testInstances.instance(i).classValue(),
								1.0))
							tp++;
						else
							fp++;

					}
					break;
				} else if (curr.isNodeFreezed()) {
					if (curr.getFreezedNodeLabel()) {
						if (Utils.eq(testInstances.instance(i).classValue(),
								1.0))
							tp++;
						else
							fp++;

					} else {
						if (Utils.eq(testInstances.instance(i).classValue(),
								1.0))
							fn++;
						else
							tn++;
					}

					break;

				} else {
					if (curr.classifySingleInstance(testInstances.instance(i)))
						curr = curr.rightNode;
					else
						curr = curr.leftNode;
				}
			}
		}
		ConfusionMatrix c = new ConfusionMatrix(new int[][] { { tn, fp },
				{ fn, tp } });
		// c.display();
		LOGGER.log(Level.INFO, tn + " " + fp);
		LOGGER.log(Level.INFO, fn + " " + tp);

	}

	public void build() {
		MartiNode currentNode = root;
		for (int level = 1; level < numLevels; level++) {
			currentNode = addLevel(currentNode);
			if (level == numLevels - 1)
				setLevelAsFinal(currentNode);
			else {
				if (!buildLevel(currentNode))
					break;
				displayLevel(currentNode);
			}
		}
		evaluateLevel(currentNode);
	}

	private void displayLevel(MartiNode node) {
		int level = node.getNodeLevel();
		for (int i = 0; i <= level; i++) {
			LOGGER.log(Level.INFO, "-----------------------------------------");
			node.display();
			LOGGER.log(Level.INFO, "Displaying global confusion matrix :");
			gb_matrix.display();
			LOGGER.log(Level.INFO, "-----------------------------------------");
			node = node.getNextNode();
		}
	}

	/**
	 * @param args
	 */
	// public static void main(String[] args) throws Exception {
	//
	// MartiBoost mBoost = null;
	// FileWriter fstream = null;
	// Optimizer.gridSearch(args[0], args[1], args[2]);
	// boolean var = true;
	// System.out.println("Entering while loop");
	// while (var) {
	// ;
	// }
	// for (int k = 1; k <= 5; k++) {
	// try {
	// fstream = new FileWriter("ionosphere-tuning-no-freezing.txt",
	// true);
	// } catch (IOException e) {
	// // TODO Auto-generated catch block
	// e.printStackTrace();
	// }
	// BufferedWriter out = new BufferedWriter(fstream);
	// try {
	// out.append("Working on Split: " + k + "\n");
	// out.append("------------------------------------------------\n");
	// out.close();
	// } catch (IOException e) {
	// // TODO Auto-generated catch block
	// e.printStackTrace();
	// }
	//
	// for (int j = 1; j <= 3; j++) {
	// TRAIN_FILE = "ionosphere-train-" + k + j + ".arff";
	// HOLDOUT_FILE = "ionosphere-holdout-" + k + j + ".arff";
	// TEST_FILE = "ionosphere-test-" + k + ".arff";
	// for (int i = 2; i <= 14; i++) {
	// try {
	//
	// // System.out.println("levels......"+i);
	// mBoost = new MartiBoost(TRAIN_FILE, HOLDOUT_FILE, i);
	//
	// } catch (Exception e1) {
	// LOGGER.log(Level.SEVERE, e1.getMessage());
	// e1.printStackTrace();
	// }
	// mBoost.build();
	// }
	// }
	// }
	// }

	public static void main(String[] args) {
		MartiBoost mBoost = null;
		try {
			long start = System.currentTimeMillis();
			mBoost = new MartiBoost(TRAIN_FILE, HOLDOUT_FILE, 4);
			mBoost.build();
			long end = System.currentTimeMillis();
			System.out.println("Elapsed time : " + (end - start));
			// mBoost.evaluateTestData(mBoost.getRoot(), TEST_FILE);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public MartiNode addLevel(MartiNode node) {
		int level = node.getNodeLevel();
		int number = node.getNodeNumber();
		MartiNode left = null, right = null;
		if (number != 0) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_NOTFIRST);
			return null;
		}
		if (node.getIsFinal()) {
			LOGGER.log(Level.WARNING, "Final node... cannot add level");
			return null;
		}
		for (int i = 0; i <= level; i++) {
			LOGGER.log(Level.INFO, "Working with node v" + node.getNodeLevel()
					+ "," + node.getNodeNumber() + "\n\n");
			if (i == 0) {
				left = node.addLeftNode();
				LOGGER.log(Level.INFO,
						"Added left node = v" + left.getNodeLevel() + ","
								+ left.getNodeNumber());
			}
			right = node.addRightNode();
			LOGGER.log(
					Level.INFO,
					"Added right node = v" + right.getNodeLevel() + ","
							+ right.getNodeNumber());
			if (node.getNodeNumber() != level) {
				if (i == 0)
					node = node.rightParent.rightNode;
				else
					node = right.leftParent.rightParent.rightNode;

				LOGGER.log(Level.INFO, "Moving to Node v" + node.getNodeLevel()
						+ "," + node.getNodeNumber());
				LOGGER.log(Level.INFO,
						"Assigning left node as v" + right.getNodeLevel() + ","
								+ right.getNodeNumber());
				node.leftNode = right;
				right.rightParent = node;
			}
		}
		return left;
	}

	public boolean buildLevel(MartiNode node) {
		boolean blnBuild = false;
		MartiNode temp = node;
		if (node == null)
			return false;
		int n = node.getNodeNumber();
		int l = node.getNodeLevel();
		ConfusionMatrix mat = null;
		if (node.getIsFinal()) {
			LOGGER.log(Level.WARNING, "the final level cannot be built");
			return false;
		}
		if (n != 0) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_NOTFIRST);
			return false;
		} else {
			if (l == 0) {
				node.train();
				node.classifyTest();
				node.classifyTrain();
				blnBuild = true;
			} else {
				for (int i = 0; i <= l; i++) {
					LOGGER.log(Level.INFO,
							"current node : " + node.getNodeLevel() + ", "
									+ node.getNodeNumber());
					node.setInstancesFromParent();
					node.setTargetGamma();

					if (node.isTrain()) {
						node.train();
						node.classifyTrain();
						node.classifyTest();
						node.computeProbabilities();
						blnBuild = true;
					} else {
						mat = node.freezeEmptyNode();
						gb_matrix.addMatrix(mat);
					}
					node = node.getNextNode();
				}
			}
		}

		freezeNodesAtALevel(temp, MetaConstants.EPSILON);
		return blnBuild;
	}

	public void evaluateLevel(MartiNode node) {

		if (node == null)
			return;
		int n = node.getNodeNumber();
		int l = node.getNodeLevel();
		int m = l / 2;
		int truenegative = 0, truepositive = 0, falsenegative = 0, falsepositive = 0;
		;
		if (n != 0) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_NOTFIRST);
			return;
		} else {
			if (l == 0) {
				LOGGER.log(Level.INFO, node.getHoldoutConfusionMatrix().getTn()
						+ " " + node.getHoldoutConfusionMatrix().getFp());
				LOGGER.log(Level.INFO, node.getHoldoutConfusionMatrix().getFn()
						+ " " + node.getHoldoutConfusionMatrix().getTp());
				return;
			}
			MartiNode parent = null;
			ConfusionMatrix conf = null;
			while (null != node) {
				parent = node.leftParent;
				if (null != parent && !parent.isNodeFreezed()) {
					conf = parent.getHoldoutConfusionMatrix();
					if (node.getNodeNumber() <= m) {
						truenegative += conf.getFp();
						falsenegative += conf.getTp();
					} else {
						truepositive += conf.getTp();
						falsepositive += conf.getFp();
					}
				}
				parent = node.rightParent;
				if (null != parent && !parent.isNodeFreezed()) {
					conf = parent.getHoldoutConfusionMatrix();
					if (node.getNodeNumber() <= m) {
						truenegative += conf.getTn();
						falsenegative += conf.getFn();
					} else {
						truepositive += conf.getFn();
						falsepositive += conf.getTn();
					}
				}
				node = node.getNextNode();
			}
			truenegative += gb_matrix.getTn();
			falsepositive += gb_matrix.getFp();
			falsenegative += gb_matrix.getFn();
			truepositive += gb_matrix.getTp();
			LOGGER.log(Level.INFO, truenegative + " " + falsepositive);
			LOGGER.log(Level.INFO, falsenegative + " " + truepositive);
			LOGGER.log(Level.INFO, "Acc+: " + truepositive * 1.0f
					/ (truepositive + falsenegative));
			LOGGER.log(Level.INFO, "Acc-: " + truenegative * 1.0f
					/ (truenegative + falsepositive));
			LOGGER.log(
					Level.INFO,
					"Acc : "
							+ (truepositive + truenegative)
							* 1.0f
							/ (truepositive + falsenegative + truenegative + falsepositive));
		}
	}

	public void setLevelAsFinal(MartiNode node) {
		int l = node.getNodeLevel();
		int n = node.getNodeNumber();
		if (n != 0) {
			LOGGER.log(Level.WARNING, MessageConstants.WARN_NODE_NOTFIRST);
			return;
		} else {
			for (int i = 0; i <= l; i++) {
				node.setFinal();
				node = node.getNextNode();
			}
		}

	}
}
