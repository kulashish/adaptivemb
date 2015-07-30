package in.ac.iitb.cse.qh.marti;

import in.ac.iitb.cse.qh.data.ConfusionMatrix;
import in.ac.iitb.cse.qh.util.WekaUtil;

import java.util.logging.Level;
import java.util.logging.Logger;

import weka.core.Instances;

public class AdaptiveMartiBoost {
	private static final Logger LOGGER = Logger
			.getLogger(AdaptiveMartiBoost.class.getName());

	private static final String TRAIN_FILE = "/Users/ashish/meta/dataset/sonar-train-53.arff";
	private static final String HOLDOUT_FILE = "/Users/ashish/meta/dataset/sonar-holdout-53.arff";
	private static final String TEST_FILE = "/Users/ashish/meta/dataset/sonar-test-5.arff";
	private AdaptiveMartiLevel rootLevel;
	private int numLevels;
	private int numNodes;
	private ConfusionMatrix gb_matrix;

	static {
		LOGGER.setLevel(Level.INFO);
	}

	public AdaptiveMartiBoost() {

	}

	public AdaptiveMartiBoost(String trainData, String holdoutData, int levels)
			throws Exception {
		this.numLevels = levels;
		Instances trainInstances = WekaUtil.getInstances(trainData);
		Instances holdoutInstances = WekaUtil.getInstances(holdoutData);
		AdaptiveMartiNodeFactory.setInstances(trainInstances);
		rootLevel = new AdaptiveMartiLevel(0);
		rootLevel.addNode(new AdaptiveMartiNode(trainInstances,
				holdoutInstances, rootLevel, 0));
		// rootLevel.build();
		// root.display();
		gb_matrix = new ConfusionMatrix();
		LOGGER.log(Level.INFO, "Initial global conf matrix :");
		gb_matrix.display();
	}

	// public AdaptiveMartiBoost(AdaptiveMartiNode root, int levels,
	// ConfusionMatrix mat) {
	// this.root = root;
	// this.numLevels = levels;
	// this.gb_matrix = mat;
	// }

	public AdaptiveMartiLevel getRootLevel() {
		return rootLevel;
	}

	public int getNumLevels() {
		return numLevels;
	}

	public void build() throws Exception {
		AdaptiveMartiLevel currentLevel = rootLevel;
		boolean blnFinal = false;
		while (currentLevel.getLevelNumber() < numLevels - 1) {
			LOGGER.log(Level.INFO,
					"Building level : " + currentLevel.getLevelNumber());
			currentLevel.build();
			numNodes += currentLevel.getNodes().size();
			currentLevel.evaluateLevel(blnFinal);
			currentLevel = currentLevel.getNextLevel();
		}
		currentLevel.evaluateLevel(blnFinal = true);
	}

	public void evaluateTestData(String testFile) throws Exception {
		rootLevel.getNodes().get(0).getInData()
				.setTestInstances(WekaUtil.getInstances(testFile));
		AdaptiveMartiLevel currentLevel = rootLevel;
		while (currentLevel.getLevelNumber() < numLevels) {
			System.out.println("Level " + currentLevel.getLevelNumber());
			if (currentLevel.getLevelNumber() == numLevels - 1)
				currentLevel.setFinalLevel(true);
			currentLevel.classifyRouteTestInstances(gb_matrix);
			currentLevel = currentLevel.getNextLevel();
		}
		gb_matrix.display();
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		AdaptiveMartiBoost mBoost = null;
		try {
			mBoost = new AdaptiveMartiBoost(TRAIN_FILE, HOLDOUT_FILE, 15);
			mBoost.build();
			LOGGER.log(Level.INFO, "--- Number of nodes in the adaptive MB: "
					+ mBoost.numNodes);
			LOGGER.log(Level.INFO, "--- Evaluating test instances ---");
			mBoost.evaluateTestData(TEST_FILE);
		} catch (Exception e1) {
			LOGGER.log(Level.SEVERE, e1.getMessage());
			e1.printStackTrace();
		}
	}

}
